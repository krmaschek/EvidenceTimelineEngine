"""Pipeline for one case: documents -> batches -> LLM -> checks -> merge -> sorted timeline.

The CLI and the Temporal workflow share `plan_case` and `build_run`. The rest is the
CLI's way of running a case without Temporal: `process_batch` and `decide_pair`
follow the same rules as the workflow's `run_batch` and `match_pair`.
"""

import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import Literal

import httpx

from evidence_timeline import merge
from evidence_timeline.batching import build_batches, check_coverage
from evidence_timeline.datasets import load_domain
from evidence_timeline.documents import load_documents
from evidence_timeline.extractors import EventExtractor, ExtractionError
from evidence_timeline.merge import EventMatcher
from evidence_timeline.models import (
    Batch,
    BatchOutcome,
    BatchReport,
    CasePlan,
    Domain,
    EventPair,
    ExtractorInfo,
    PairDecision,
    TimelineRun,
)
from evidence_timeline.prompts import prompt_sha256
from evidence_timeline.timeline import ORDERING_NOTE, build_timeline_events, sort_timeline

NOTES = [
    "Events that more than one document records are merged into one, keeping every quote. "
    "Two events are only compared when they are the same type and status and their dates are "
    f"within {merge.WINDOW.days} days, so the same event recorded much later is not found.",
    "When merged sources disagree about the day, the event is marked 'conflicting' and every "
    "date is kept. The disagreement is reported, not resolved.",
    "An empty citation_errors list means each quote was found at its cited lines. "
    "It does not mean the quote supports the event.",
    ORDERING_NOTE,
]


def plan_case(case_dir: Path, max_chars: int, extractor: ExtractorInfo) -> CasePlan:
    """Read one case and divide it into batches. No model is called here.

    The domain comes from the case's dataset folder, so each case is extracted with
    its own dataset's definition of an event.
    """
    case_id = case_dir.name.removeprefix("case_")
    documents = load_documents(case_dir)
    batches = build_batches(case_id, documents, max_chars)
    check_coverage(documents, batches)
    return CasePlan(
        case_id=case_id,
        domain=load_domain(case_dir),
        extractor=extractor,
        batches=batches,
        document_sha256={document.document_id: document.sha256 for document in documents},
        total_lines=sum(len(document.lines) for document in documents),
    )


async def run_case(
    case_dir: Path, extractor: EventExtractor, max_chars: int, matcher: EventMatcher | None = None
) -> TimelineRun:
    plan = plan_case(case_dir, max_chars, extractor.info)

    coroutines = [process_batch(batch, plan.domain, extractor) for batch in plan.batches]
    outcomes = await asyncio.gather(*coroutines)  # results come back in the order of the batches

    events = [event for outcome in outcomes for event in outcome.events]
    questions = [decide_pair(pair, matcher) for pair in merge.candidates(events)]
    answers = await asyncio.gather(*questions)
    decisions = [decision for decision in answers if decision is not None]

    return build_run(
        plan,
        outcomes,
        decisions,
        max_chars,
        run_id=uuid.uuid4().hex,
        created_at=dt.datetime.now(dt.UTC),
    )


async def process_batch(batch: Batch, domain: Domain, extractor: EventExtractor) -> BatchOutcome:
    """Extract one batch.

    The extractor retries on its own; if it still fails, the batch is recorded as failed.
    """
    try:
        result = await extractor.extract_events(batch, domain)
    except ExtractionError as error:
        # Only expected extraction failures are caught. Any other exception is a bug and stops the run.
        failed = BatchReport(
            batch_id=batch.batch_id,
            lines=batch.line_labels(),
            status="failed",
            attempts=error.attempts,
            error=str(error),
        )
        return BatchOutcome(report=failed, events=[])

    succeeded = BatchReport(
        batch_id=batch.batch_id,
        lines=batch.line_labels(),
        status="succeeded",
        attempts=result.attempts,
        model=result.model,
        usage=result.usage,
    )
    return BatchOutcome(report=succeeded, events=build_timeline_events(batch, result.events))


async def decide_pair(pair: EventPair, matcher: EventMatcher | None) -> PairDecision | None:
    """Ask whether two events are the same. None means there is no answer, so they stay separate."""
    if matcher is None:  # no matcher (a fake run), so nothing is merged
        return None
    try:
        decision = await matcher.is_same(pair)
    except (TimeoutError, httpx.HTTPError, ValueError, KeyError, IndexError):
        return None  # could not decide: a duplicate is better than a wrong merge
    return PairDecision(a=pair.a.event_id, b=pair.b.event_id, same_event=decision.same_event, reason=decision.reason)


def build_run(
    plan: CasePlan,
    outcomes: list[BatchOutcome],
    decisions: list[PairDecision],
    max_chars: int,
    run_id: str,
    created_at: dt.datetime,
) -> TimelineRun:
    """Combine the finished batches into one run. The id and time are passed in because a workflow
    must get them from Temporal."""
    reports = [outcome.report for outcome in outcomes]
    events = [event for outcome in outcomes for event in outcome.events]
    merged = merge.merge_events(events, decisions)

    lines_in_successful_batches = sum(
        batch.line_count for batch, report in zip(plan.batches, reports) if report.status == "succeeded"
    )

    failed_batches = sum(report.status == "failed" for report in reports)
    status: Literal["completed", "partial", "failed"]
    if failed_batches == 0:
        status = "completed"
    elif failed_batches < len(reports):
        status = "partial"
    else:
        status = "failed"

    return TimelineRun(
        run_id=run_id,
        created_at=created_at,
        case_id=plan.case_id,
        domain=plan.domain.name,
        status=status,
        notes=NOTES,
        extractor=plan.extractor,
        max_chars_per_batch=max_chars,
        prompt_sha256=prompt_sha256(plan.domain),
        document_sha256=plan.document_sha256,
        total_lines=plan.total_lines,
        lines_in_successful_batches=lines_in_successful_batches,
        batches=reports,
        events=sort_timeline(merged),
        pair_decisions=decisions,
    )
