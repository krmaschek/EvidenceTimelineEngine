"""What happens for one case, and in what order.

Temporal replays a workflow from its history after a crash, so this code must make
the same decisions every time. Real work (files, LLM calls, database) happens in
activities, and the run's id and time come from Temporal.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

# Let the sandbox reuse these modules instead of re-importing them for every run;
# they have no side effects on import.
with workflow.unsafe.imports_passed_through():
    from evidence_timeline import merge
    from evidence_timeline.activities import Activities
    from evidence_timeline.models import (
        Batch,
        BatchOutcome,
        BatchReport,
        CaseRequest,
        EventPair,
        TimelineRun,
    )
    from evidence_timeline.pipeline import build_run

# Temporal handles timeouts and retries. The extractor's own retry loop is off in the worker,
# so the two don't multiply.
BATCH_TIMEOUT = timedelta(seconds=120)
MATCH_TIMEOUT = timedelta(seconds=60)  # one pair is a much smaller question than one batch
LLM_RETRIES = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)
# Without a policy Temporal retries forever. A missing folder or a bad plan won't fix itself,
# so these errors fail at once.
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
        # Start every batch at once; the worker limits how many actually run.
        coroutines = [self.run_batch(batch) for batch in plan.batches]
        outcomes = await asyncio.gather(*coroutines)

        # Find duplicates: simple rules pick the pairs worth checking, and the LLM decides each one.
        events = [event for outcome in outcomes for event in outcome.events]
        questions = [self.match_pair(pair) for pair in merge.candidates(events)]
        answers = await asyncio.gather(*questions)
        confirmed_pairs = [pair for pair in answers if pair is not None]

        run = build_run(
            plan,
            outcomes,
            confirmed_pairs,
            request.max_chars,
            run_id=workflow.uuid4().hex,  # from Temporal, so a replay produces the same run
            created_at=workflow.now(),
        )
        await workflow.execute_activity_method(
            Activities.save_timeline_run, run, start_to_close_timeout=timedelta(seconds=60)
        )
        return run

    async def run_batch(self, batch: Batch) -> BatchOutcome:
        """Extract one batch on a worker. If every attempt fails, the batch is recorded as failed."""
        try:
            return await workflow.execute_activity_method(
                Activities.extract_batch,
                batch,
                start_to_close_timeout=BATCH_TIMEOUT,
                retry_policy=LLM_RETRIES,
            )
        except ActivityError as error:
            # Out of attempts. The other batches carry on and the run ends up "partial".
            failed = BatchReport(
                batch_id=batch.batch_id,
                lines=batch.line_labels(),
                status="failed",
                attempts=LLM_RETRIES.maximum_attempts,
                error=str(error.cause),
            )
            return BatchOutcome(report=failed, events=[])

    async def match_pair(self, pair: EventPair) -> EventPair | None:
        """Ask a worker whether two events are the same. None means they stay separate."""
        try:
            decision = await workflow.execute_activity_method(
                Activities.match_pair,
                pair,
                start_to_close_timeout=MATCH_TIMEOUT,
                retry_policy=LLM_RETRIES,
            )
        except ActivityError:
            # A pair we could not decide is left alone. A duplicate is better than a wrong merge.
            return None
        return pair if decision.same_event else None
