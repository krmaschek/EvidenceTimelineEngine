"""Pipeline for one case: documents -> batches -> LLM -> checks -> sorted timeline.

The work is split into three pieces so that both ways of running it, the CLI and
the Temporal workflow, use the same code:

- `plan_case` reads the files and builds the batches;
- `process_batch` turns one batch into events;
- `build_run` puts the finished parts together.

`run_case` is the CLI's way: it starts all batches together and lets the extractor
limit how many LLM calls run at the same time. A failed batch is recorded and the
others still finish, but the run is then marked "partial" (or "failed").
"""

import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import Literal

from evidence_timeline.batching import build_batches, check_coverage
from evidence_timeline.documents import load_documents
from evidence_timeline.extractors import EventExtractor, ExtractionError
from evidence_timeline.models import (
    Batch,
    BatchOutcome,
    BatchReport,
    CasePlan,
    ExtractorInfo,
    TimelineRun,
)
from evidence_timeline.prompts import PROMPT_SHA256
from evidence_timeline.timeline import ORDERING_NOTE, build_timeline_events, sort_timeline

NOTES = [
    "PRELIMINARY timeline: duplicates are not merged and conflicts between documents are not "
    "detected yet, so the same event can appear more than once.",
    "An empty citation_errors list means each quote was found at its cited lines. "
    "It does not mean the quote supports the event.",
    ORDERING_NOTE,
]


def plan_case(case_dir: Path, max_chars: int, extractor: ExtractorInfo) -> CasePlan:
    """Read one case and divide it into batches. No model is called here."""
    case_id = case_dir.name.removeprefix("case_")
    documents = load_documents(case_dir)
    batches = build_batches(case_id, documents, max_chars)
    check_coverage(documents, batches)
    return CasePlan(
        case_id=case_id,
        extractor=extractor,
        batches=batches,
        document_sha256={document.document_id: document.sha256 for document in documents},
        total_lines=sum(len(document.lines) for document in documents),
    )


async def run_case(case_dir: Path, extractor: EventExtractor, max_chars: int) -> TimelineRun:
    plan = plan_case(case_dir, max_chars, extractor.info)

    coroutines = [process_batch(batch, extractor) for batch in plan.batches]
    outcomes = await asyncio.gather(*coroutines)  # results come back in the order of the batches

    return build_run(
        plan,
        outcomes,
        max_chars,
        run_id=uuid.uuid4().hex,
        created_at=dt.datetime.now(dt.UTC),
    )


async def process_batch(batch: Batch, extractor: EventExtractor) -> BatchOutcome:
    """One batch, with its own retries inside the extractor. A failure becomes a failed report."""
    try:
        result = await extractor.extract_events(batch)
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


def build_run(
    plan: CasePlan,
    outcomes: list[BatchOutcome],
    max_chars: int,
    run_id: str,
    created_at: dt.datetime,
) -> TimelineRun:
    """Put the finished batches together. Pure: the run's identity and time are given, not made here,
    because a Temporal workflow has to take them from Temporal."""
    reports = [outcome.report for outcome in outcomes]
    events = [event for outcome in outcomes for event in outcome.events]

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
        status=status,
        notes=NOTES,
        extractor=plan.extractor,
        max_chars_per_batch=max_chars,
        prompt_sha256=PROMPT_SHA256,
        document_sha256=plan.document_sha256,
        total_lines=plan.total_lines,
        lines_in_successful_batches=lines_in_successful_batches,
        batches=reports,
        events=sort_timeline(events),
    )
