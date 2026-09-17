"""Phase 1 pipeline for one case: documents -> batches -> LLM -> checks -> sorted timeline.

Batches are processed one after another. A failed batch is recorded and the
others still run, but the run is then marked "partial" (or "failed"), never
"completed".
"""

import datetime as dt
import uuid
from pathlib import Path
from typing import Literal

from evidence_timeline.batching import build_batches, check_coverage
from evidence_timeline.documents import load_documents
from evidence_timeline.extractors import EventExtractor, ExtractionError
from evidence_timeline.models import BatchReport, TimelineEvent, TimelineRun
from evidence_timeline.prompts import PROMPT_SHA256
from evidence_timeline.timeline import ORDERING_NOTE, build_timeline_events, sort_timeline

NOTES = [
    "PRELIMINARY timeline: duplicates are not merged and conflicts between documents are not "
    "detected yet, so the same event can appear more than once.",
    "An empty citation_errors list means each quote was found at its cited lines. "
    "It does not mean the quote supports the event.",
    ORDERING_NOTE,
]


def run_case(case_dir: Path, extractor: EventExtractor, max_chars: int) -> TimelineRun:
    case_id = case_dir.name.removeprefix("case_")
    documents = load_documents(case_dir)
    batches = build_batches(case_id, documents, max_chars)
    check_coverage(documents, batches)

    reports = []
    events: list[TimelineEvent] = []
    lines_in_successful_batches = 0
    for batch in batches:
        lines = [f"{span.document_id}:{span.first_line}-{span.last_line}" for span in batch.spans]
        try:
            result = extractor.extract(batch)
        except ExtractionError as error:
            # Only expected extraction failures are caught. Any other exception is a bug and stops the run.
            reports.append(
                BatchReport(
                    batch_id=batch.batch_id,
                    lines=lines,
                    status="failed",
                    attempts=error.attempts,
                    error=str(error),
                )
            )
            continue

        reports.append(
            BatchReport(
                batch_id=batch.batch_id,
                lines=lines,
                status="succeeded",
                attempts=result.attempts,
                model=result.model,
                usage=result.usage,
            )
        )
        events += build_timeline_events(batch, result.events)
        lines_in_successful_batches += sum(len(span.lines) for span in batch.spans)

    failed = sum(report.status == "failed" for report in reports)
    status: Literal["completed", "partial", "failed"]
    if failed == 0:
        status = "completed"
    elif failed < len(reports):
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
