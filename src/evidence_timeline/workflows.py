"""What happens for one case, and in what order.

Temporal replays a workflow from its history after a crash, so this code must make
the same decisions every time. Real work (files, LLM calls, database) happens in
activities, and the run's id and time come from Temporal.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, RetryState

# Let the sandbox reuse these modules instead of re-importing them for every run;
# they have no side effects on import.
with workflow.unsafe.imports_passed_through():
    from evidence_timeline import merge
    from evidence_timeline.activities import Activities
    from evidence_timeline.models import (
        Batch,
        BatchOutcome,
        BatchReport,
        BatchTask,
        CaseRequest,
        Domain,
        EventPair,
        PairDecision,
        Progress,
        Stage,
        TimelineRun,
    )
    from evidence_timeline.pipeline import build_run

# Temporal handles timeouts and retries. The extractor's own retry loop is off in the worker,
# so the two don't multiply.
BATCH_TIMEOUT = timedelta(seconds=1200)  # a model that reasons first can need over 10 minutes for a whole document
MATCH_TIMEOUT = timedelta(seconds=120)  # one pair is a much smaller question than one batch
LLM_RETRIES = RetryPolicy(
    initial_interval=timedelta(seconds=15),  # a rate limit (HTTP 429) can last minutes
    backoff_coefficient=2.0,  # then 30 s, 60 s and 120 s
    maximum_attempts=5,
    non_retryable_error_types=["PermanentExtractionError"],  # e.g. a wrong API key
)
# Without a policy Temporal retries forever. A missing folder or file, or a bad plan, won't fix
# itself, so these errors fail at once.
PLAN_RETRIES = RetryPolicy(
    maximum_attempts=3, non_retryable_error_types=["ValueError", "RuntimeError", "FileNotFoundError"]
)


@workflow.defn
class BuildTimelineWorkflow:
    def __init__(self) -> None:
        # What the progress query reports. Only the workflow itself changes these.
        self.stage: Stage = "planning"
        self.total_batches = 0
        self.finished_batches = 0

    @workflow.query
    def progress(self) -> Progress:
        """Answered by a worker whenever the API asks. It only reads, never changes anything."""
        return Progress(stage=self.stage, total_batches=self.total_batches, finished_batches=self.finished_batches)

    @workflow.run
    async def run(self, request: CaseRequest) -> TimelineRun:
        plan = await workflow.execute_activity_method(
            Activities.plan_case,
            request,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=PLAN_RETRIES,
        )
        # Start every batch at once; the worker limits how many actually run.
        self.stage = "extracting"
        self.total_batches = len(plan.batches)
        coroutines = [self.run_batch(batch, plan.domain) for batch in plan.batches]
        outcomes = await asyncio.gather(*coroutines)

        # Find duplicates: simple rules pick the pairs worth checking, and the LLM decides each one.
        self.stage = "merging"
        events = [event for outcome in outcomes for event in outcome.events]
        questions = [self.match_pair(pair) for pair in merge.candidates(events)]
        answers = await asyncio.gather(*questions)
        decisions = [decision for decision in answers if decision is not None]

        run = build_run(
            plan,
            outcomes,
            decisions,
            request.max_chars,
            run_id=workflow.uuid4().hex,  # from Temporal, so a replay produces the same run
            created_at=workflow.now(),
        )
        self.stage = "saving"
        await workflow.execute_activity_method(
            Activities.save_timeline_run, run, start_to_close_timeout=timedelta(seconds=60)
        )
        self.stage = "done"
        return run

    async def run_batch(self, batch: Batch, domain: Domain) -> BatchOutcome:
        """Extract one batch on a worker. If every attempt fails, the batch is recorded as failed."""
        try:
            outcome = await workflow.execute_activity_method(
                Activities.extract_batch,
                BatchTask(batch=batch, domain=domain),
                start_to_close_timeout=BATCH_TIMEOUT,
                retry_policy=LLM_RETRIES,
            )
        except ActivityError as error:
            # Out of attempts, or an error that retrying cannot fix. The other batches carry on
            # and the run ends up "partial".
            attempts = LLM_RETRIES.maximum_attempts
            if error.retry_state == RetryState.NON_RETRYABLE_FAILURE:
                attempts = 1  # a permanent error, such as a wrong API key, fails on the first try
            failed = BatchReport(
                batch_id=batch.batch_id,
                lines=batch.line_labels(),
                status="failed",
                attempts=attempts,
                error=str(error.cause),
            )
            outcome = BatchOutcome(report=failed, events=[])

        self.finished_batches += 1  # succeeded or failed, this batch is no longer running
        return outcome

    async def match_pair(self, pair: EventPair) -> PairDecision | None:
        """Ask a worker whether two events are the same. None means there is no answer, so they stay separate."""
        try:
            result = await workflow.execute_activity_method(
                Activities.match_pair,
                pair,
                start_to_close_timeout=MATCH_TIMEOUT,
                retry_policy=LLM_RETRIES,
            )
        except ActivityError:
            # A pair we could not decide is left alone. A duplicate is better than a wrong merge.
            return None
        return PairDecision(
            a=pair.a.event_id, b=pair.b.event_id, same_event=result.same_event, reason=result.reason, usage=result.usage
        )
