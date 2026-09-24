"""The workflow's steps that touch the outside world: the files, the model, the database.

They are methods of one class so that every batch shares a single extractor, and
with it one HTTP connection pool and one concurrency limit. The worker creates
that instance once, at start-up.

Nothing here catches errors on purpose: an activity that raises is retried by
Temporal, and the workflow decides what a batch that never succeeded means.
"""

from pathlib import Path

from temporalio import activity

from evidence_timeline import pipeline, storage
from evidence_timeline.extractors import EventExtractor
from evidence_timeline.merge import EventMatcher
from evidence_timeline.models import (
    BatchOutcome,
    BatchReport,
    BatchTask,
    CasePlan,
    CaseRequest,
    EventPair,
    MatchResult,
    TimelineRun,
)
from evidence_timeline.timeline import build_timeline_events


class Activities:
    def __init__(
        self, extractor: EventExtractor, matcher: EventMatcher | None, database_url: str | None
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.database_url = database_url

    @activity.defn
    async def plan_case(self, request: CaseRequest) -> CasePlan:
        """The plan also says which extractor will run, because the workflow cannot see it."""
        return pipeline.plan_case(Path(request.case_dir), request.max_chars, self.extractor.info)

    @activity.defn
    async def extract_batch(self, task: BatchTask) -> BatchOutcome:
        batch = task.batch
        result = await self.extractor.extract_events(batch, task.domain)
        report = BatchReport(
            batch_id=batch.batch_id,
            lines=batch.line_labels(),
            status="succeeded",
            attempts=activity.info().attempt,  # counted by Temporal, not by us
            model=result.model,
            usage=result.usage,
        )
        return BatchOutcome(report=report, events=build_timeline_events(batch, result.events))

    @activity.defn
    async def match_pair(self, pair: EventPair) -> MatchResult:
        if self.matcher is None:  # a fake run has no model to ask, so nothing is merged
            return MatchResult(same_event=False, reason="no matcher configured")
        return await self.matcher.is_same(pair)

    @activity.defn
    async def save_timeline_run(self, run: TimelineRun) -> None:
        """Writing the same run twice overwrites the same rows, which is what makes a retry safe."""
        if self.database_url:
            await storage.save_run(run, self.database_url)
