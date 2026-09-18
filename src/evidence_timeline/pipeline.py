"""Pipeline for one case: documents -> batches -> LLM -> checks -> sorted timeline.

All batches are started together; the extractor limits how many LLM calls run
at the same time. A failed batch is recorded and the others still finish, but
the run is then marked "partial" (or "failed"), never "completed".
"""

import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import Literal

from evidence_timeline.batching import build_batches, check_coverage
from evidence_timeline.documents import load_documents
from evidence_timeline.extractors import EventExtractor, ExtractionError
from evidence_timeline.models import Batch, BatchReport, TimelineEvent, TimelineRun
from evidence_timeline.prompts import PROMPT_SHA256
from evidence_timeline.timeline import ORDERING_NOTE, build_timeline_events, sort_timeline

NOTES = [
    "PRELIMINARY timeline: duplicates are not merged and conflicts between documents are not "
    "detected yet, so the same event can appear more than once.",
    "An empty citation_errors list means each quote was found at its cited lines. "
    "It does not mean the quote supports the event.",
    ORDERING_NOTE,
]


async def run_case(case_dir: Path, extractor: EventExtractor, max_chars: int) -> TimelineRun:
    case_id = case_dir.name.removeprefix("case_")
    documents = load_documents(case_dir)
    batches = build_batches(case_id, documents, max_chars)
    check_coverage(documents, batches)

    # gather() returns the results in the same order as the batches.
    coroutines = [process_batch(batch, extractor) for batch in batches]
    results = await asyncio.gather(*coroutines)

    reports: list[BatchReport] = []
    events: list[TimelineEvent] = []
    for report, batch_events in results:
        reports.append(report)
        events += batch_events

    lines_in_successful_batches = sum(
        batch.line_count for batch, report in zip(batches, reports) if report.status == "succeeded"
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
        run_id=uuid.uuid4().hex,
        created_at=dt.datetime.now(dt.UTC),
        case_id=case_id,
        status=status,
        notes=NOTES,
        extractor=extractor.info,
        max_chars_per_batch=max_chars,
        prompt_sha256=PROMPT_SHA256,
        document_sha256={document.document_id: document.sha256 for document in documents},
        total_lines=sum(len(document.lines) for document in documents),
        lines_in_successful_batches=lines_in_successful_batches,
        batches=reports,
        events=sort_timeline(events),
    )


async def process_batch(batch: Batch, extractor: EventExtractor) -> tuple[BatchReport, list[TimelineEvent]]:
    lines = [f"{span.document_id}:{span.first_line}-{span.last_line}" for span in batch.spans]
    try:
        result = await extractor.extract_events(batch)
    except ExtractionError as error:
        # Only expected extraction failures are caught. Any other exception is a bug and stops the run.
        failed = BatchReport(
            batch_id=batch.batch_id, lines=lines, status="failed", attempts=error.attempts, error=str(error)
        )
        return failed, []

    succeeded = BatchReport(
        batch_id=batch.batch_id,
        lines=lines,
        status="succeeded",
        attempts=result.attempts,
        model=result.model,
        usage=result.usage,
    )
    return succeeded, build_timeline_events(batch, result.events)
