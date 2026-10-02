"""Small builders and the Temporal test setup shared by the tests."""

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from evidence_timeline.activities import Activities
from evidence_timeline.datasets import load_domain
from evidence_timeline.extractors import EventExtractor
from evidence_timeline.merge import EventMatcher
from evidence_timeline.models import (
    Batch,
    DocumentSpan,
    EvidenceQuote,
    ExtractedEvent,
    ExtractorInfo,
    TimelineEvent,
    TimelineRun,
)
from evidence_timeline.worker import TASK_QUEUE
from evidence_timeline.workflows import BuildTimelineWorkflow

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "clinical_v1"
CASE_A_DIR = DATASET_DIR / "inputs" / "case_A"
CASE_B_DIR = DATASET_DIR / "inputs" / "case_B"
GOLD_DIR = DATASET_DIR / "gold"
CLINICAL_DOMAIN = load_domain(CASE_A_DIR)  # read from the dataset's domain.json and scope.md

APPROXIMATE_MARCH = {
    "date": None,
    "date_precision": "approximate",
    "date_earliest": dt.date(2025, 3, 1),
    "date_latest": dt.date(2025, 3, 31),
}
CONFLICTING_MARCH = {
    "date": None,
    "date_precision": "conflicting",
    "alternative_dates": [dt.date(2025, 3, 12), dt.date(2025, 3, 10)],
}
NO_DATE = {"date": None, "date_precision": "unknown"}


def make_batch(lines: list[str]) -> Batch:
    return Batch(batch_id="X-batch-001", spans=[DocumentSpan(document_id="D01", first_line=1, lines=lines)])


def make_quote(
    line: int = 1, quote: str = "text", document_id: str = "D01", date_text: str | None = None
) -> EvidenceQuote:
    return EvidenceQuote(document_id=document_id, line_start=line, line_end=line, quote=quote, date_text=date_text)


def make_event(**changes: Any) -> ExtractedEvent:
    fields: dict[str, Any] = {
        "event_type": "visit",
        "status": "completed",
        "description": "Visit",
        "date": dt.date(2025, 1, 6),
        "date_precision": "exact",
        "date_earliest": None,
        "date_latest": None,
        "alternative_dates": [],
        "evidence": [make_quote()],
        "review_reasons": [],
    }
    fields.update(changes)
    return ExtractedEvent(**fields)


def make_timeline_event(event_id: str, **changes: Any) -> TimelineEvent:
    event = make_event(**changes)
    return TimelineEvent(
        **event.model_dump(), event_id=event_id, batch_id="X-batch-001", citation_errors=[], needs_review=False
    )


def make_run(events: list[TimelineEvent], kind: Any = "llm", status: Any = "completed") -> TimelineRun:
    return TimelineRun(
        run_id=uuid.uuid4().hex,
        created_at=dt.datetime.now(dt.UTC),
        case_id="R",
        status=status,
        notes=[],
        extractor=ExtractorInfo(kind=kind, model=None, settings={}),
        max_chars_per_batch=20_000,
        prompt_sha256="",
        document_sha256={},
        total_lines=0,
        lines_in_successful_batches=0,
        batches=[],
        events=events,
    )


@asynccontextmanager
async def temporal_worker(extractor: EventExtractor, matcher: EventMatcher | None = None) -> AsyncIterator[Client]:
    """Start Temporal's test server and a worker on it, and give back a client for that server.

    `start_time_skipping()` downloads the test server the first time. Without it the
    test is skipped, so the rest of the suite stays offline.
    """
    try:
        environment = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)
    except Exception as error:  # no test server, for example without a network connection
        pytest.skip(f"Temporal's test server could not start: {error}")

    async with environment:
        activities = Activities(extractor, matcher, database_url=None)
        worker = Worker(
            environment.client,
            task_queue=TASK_QUEUE,
            workflows=[BuildTimelineWorkflow],
            activities=[
                activities.plan_case,
                activities.extract_batch,
                activities.match_pair,
                activities.save_timeline_run,
            ],
        )
        async with worker:
            yield environment.client
