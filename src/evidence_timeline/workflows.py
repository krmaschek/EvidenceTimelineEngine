"""The plan: what happens for one case, and in what order.

A workflow is replayed from its history after a crash, so it must always take the
same decisions. That is why there are no files, no HTTP calls and no database
writes here, and why the run's id and time come from Temporal instead of from
`uuid4()` and `datetime.now()`.

Imports are passed through the workflow sandbox: our own modules are ordinary
Python, and importing them must not execute anything.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from evidence_timeline.activities import Activities
    from evidence_timeline.models import Batch, BatchOutcome, BatchReport, CaseRequest, TimelineRun
    from evidence_timeline.pipeline import build_run

# Temporal owns the clock and the retries here; the extractor's own loop is switched off in the worker,
# so there is one visible retry mechanism instead of two that multiply.
BATCH_TIMEOUT = timedelta(seconds=120)
BATCH_RETRIES = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)
# Without a policy Temporal retries forever. A missing case folder or a broken batch plan
# never succeeds, so those two errors fail the run instead.
PLAN_RETRIES = RetryPolicy(maximum_attempts=3, non_retryable_error_types=["ValueError", "RuntimeError"])


@workflow.defn
class BuildTimelineWorkflow:
    @workflow.run
    async def run(self, request: CaseRequest) -> TimelineRun:
        plan = await workflow.execute_activity_method(
            Activities.plan_case,
            request,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=PLAN_RETRIES,
        )
        # All batches are started together. How many really run at once is the worker's limit.
        coroutines = [self.run_batch(batch) for batch in plan.batches]
        outcomes = await asyncio.gather(*coroutines)

        run = build_run(
            plan,
            outcomes,
            request.max_chars,
            run_id=workflow.uuid4().hex,  # from Temporal, so a replay produces the same run
            created_at=workflow.now(),
        )
        await workflow.execute_activity_method(
            Activities.save_timeline_run, run, start_to_close_timeout=timedelta(seconds=60)
        )
        return run

    async def run_batch(self, batch: Batch) -> BatchOutcome:
        """Ask a worker to extract one batch, and decide what it means if it never succeeds."""
        try:
            return await workflow.execute_activity_method(
                Activities.extract_batch,
                batch,
                start_to_close_timeout=BATCH_TIMEOUT,
                retry_policy=BATCH_RETRIES,
            )
        except ActivityError as error:
            # Temporal has used up its attempts. The other batches still finish and the run is "partial".
            failed = BatchReport(
                batch_id=batch.batch_id,
                lines=batch.line_labels(),
                status="failed",
                attempts=BATCH_RETRIES.maximum_attempts,
                error=str(error.cause),
            )
            return BatchOutcome(report=failed, events=[])
