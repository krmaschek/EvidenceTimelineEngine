"""The Temporal workflow, run in Temporal's own test environment.

No server is needed: `temporal_worker` in helpers.py starts Temporal's small test
server. The test server also skips the waiting between retries, so a failing batch
costs no real time.
"""

import asyncio
from pathlib import Path

import pytest
from helpers import CASE_A_DIR, CASE_B_DIR, temporal_worker
from temporalio.client import WorkflowFailureError

from evidence_timeline.extractors import (
    EventExtractor,
    ExtractionError,
    ExtractionResult,
    FakeExtractor,
    PermanentExtractionError,
)
from evidence_timeline.merge import EventMatcher
from evidence_timeline.models import Batch, CaseRequest, Domain, EventPair, MatchDecision, TimelineRun
from evidence_timeline.worker import TASK_QUEUE
from evidence_timeline.workflows import BuildTimelineWorkflow


class FailingExtractor:
    """Fails the given batches and behaves like the fake extractor for the rest."""

    info = FakeExtractor.info

    def __init__(self, failing_batch_ids: set[str]) -> None:
        self.failing_batch_ids = failing_batch_ids

    async def extract_events(self, batch: Batch, domain: Domain) -> ExtractionResult:
        if batch.batch_id in self.failing_batch_ids:
            raise ExtractionError("provider down", attempts=1)
        return await FakeExtractor().extract_events(batch, domain)


class WrongKeyExtractor:
    """Fails every batch with an error that retrying cannot fix, and counts how often it is called."""

    info = FakeExtractor.info

    def __init__(self) -> None:
        self.calls = 0

    async def extract_events(self, batch: Batch, domain: Domain) -> ExtractionResult:
        self.calls += 1
        raise PermanentExtractionError("HTTP 401: wrong API key", attempts=1)


class AlwaysTheSameMatcher:
    """Says yes to every pair, so the tests can see the merging without a model."""

    async def is_same(self, pair: EventPair) -> MatchDecision:
        return MatchDecision(same_event=True, reason="test matcher")


async def run_workflow(
    extractor: EventExtractor, case_dir: Path, max_chars: int, matcher: EventMatcher | None = None
) -> TimelineRun:
    """Start a test server with a worker, and carry out one case."""
    async with temporal_worker(extractor, matcher) as client:
        return await client.execute_workflow(
            BuildTimelineWorkflow.run,
            CaseRequest(case_dir=str(case_dir), max_chars=max_chars),
            id=f"test-{case_dir.name}",
            task_queue=TASK_QUEUE,
        )


def test_the_workflow_builds_a_complete_run():
    run = asyncio.run(run_workflow(FakeExtractor(), CASE_A_DIR, max_chars=500))  # 4 batches

    assert run.status == "completed"
    assert run.case_id == "A"
    assert run.extractor.kind == "fake"
    assert run.total_lines == run.lines_in_successful_batches == 41
    assert [batch.batch_id for batch in run.batches] == [f"A-batch-00{n}" for n in (1, 2, 3, 4)]
    assert [batch.attempts for batch in run.batches] == [1, 1, 1, 1]  # counted by Temporal
    assert [event.evidence[0].document_id for event in run.events] == ["A01", "A02", "A03", "A04"]
    assert all(event.citation_errors == [] for event in run.events)


def test_a_batch_that_never_succeeds_makes_the_run_partial():
    run = asyncio.run(run_workflow(FailingExtractor({"B-batch-002"}), CASE_B_DIR, max_chars=900))  # 3 batches

    assert run.status == "partial"
    assert [batch.status for batch in run.batches] == ["succeeded", "failed", "succeeded"]

    failed = run.batches[1]
    assert failed.attempts == 3  # Temporal gave up after LLM_RETRIES
    assert "provider down" in failed.error
    assert failed.lines == ["B03:1-11", "B04:1-10"]
    assert {event.batch_id for event in run.events} == {"B-batch-001", "B-batch-003"}


def test_an_error_that_retrying_cannot_fix_is_tried_only_once():
    extractor = WrongKeyExtractor()
    run = asyncio.run(run_workflow(extractor, CASE_A_DIR, max_chars=500))  # 4 batches

    assert run.status == "failed"
    assert extractor.calls == 4  # one call per batch, no retries
    assert [batch.attempts for batch in run.batches] == [1, 1, 1, 1]
    assert "wrong API key" in run.batches[0].error


def test_a_missing_case_folder_fails_instead_of_retrying_forever():
    missing = CASE_A_DIR.parent / "case_missing"

    with pytest.raises(WorkflowFailureError) as error:
        asyncio.run(run_workflow(FakeExtractor(), missing, max_chars=500))

    # WorkflowFailureError -> ActivityError -> the ValueError the activity raised
    assert "No Markdown documents found" in str(error.value.cause.cause)


def test_a_worker_with_a_matcher_merges_what_it_confirms():
    # The fake extractor returns one placeholder event per document, all undated and all of the
    # same kind, so every pair is a candidate and this matcher confirms all of them.
    run = asyncio.run(run_workflow(FakeExtractor(), CASE_A_DIR, 500, AlwaysTheSameMatcher()))

    assert len(run.events) == 1
    assert len(run.events[0].merged_from) == 4
    assert len(run.pair_decisions) == 6  # every pair of the four events, each with its answer
    assert [quote.document_id for quote in run.events[0].evidence] == ["A01", "A02", "A03", "A04"]
