import json

import pytest
from helpers import CASE_A_DIR, CASE_B_DIR, DATASET_DIR, REPO_ROOT, make_event, make_quote

from evidence_timeline.extractors import ExtractionError, ExtractionResult, FakeExtractor
from evidence_timeline.models import Batch, TimelineRun
from evidence_timeline.pipeline import run_case


class FailingExtractor:
    """Fails the given batches and behaves like the fake extractor for the rest."""

    info = FakeExtractor.info

    def __init__(self, failing_batch_ids: set[str]) -> None:
        self.failing_batch_ids = failing_batch_ids

    def extract(self, batch: Batch) -> ExtractionResult:
        if batch.batch_id in self.failing_batch_ids:
            raise ExtractionError("provider down", attempts=3)
        return FakeExtractor().extract(batch)


def test_fake_run_covers_every_line():
    run = run_case(CASE_A_DIR, FakeExtractor(), max_chars=20_000)

    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert run.status == "completed"
    assert run.case_id == "A"
    assert run.extractor.kind == "fake"
    assert run.total_lines == run.lines_in_successful_batches == 41
    assert run.document_sha256 == {d["document_id"]: d["sha256"] for d in manifest["documents"] if d["case_id"] == "A"}
    assert [event.evidence[0].document_id for event in run.events] == ["A01", "A02", "A03", "A04"]
    assert all(event.citation_errors == [] for event in run.events)
    assert "PRELIMINARY" in run.notes[0]
    assert TimelineRun.model_validate_json(run.model_dump_json()) == run


def test_a_failed_batch_makes_the_run_partial():
    run = run_case(CASE_B_DIR, FailingExtractor({"B-batch-002"}), max_chars=900)

    assert run.status == "partial"
    assert [(batch.batch_id, batch.status, batch.error) for batch in run.batches] == [
        ("B-batch-001", "succeeded", None),
        ("B-batch-002", "failed", "provider down"),
        ("B-batch-003", "succeeded", None),
    ]
    assert run.batches[1].lines == ["B03:1-11", "B04:1-10"]
    assert run.lines_in_successful_batches == run.total_lines - 21
    assert {event.batch_id for event in run.events} == {"B-batch-001", "B-batch-003"}


def test_a_run_without_any_successful_batch_is_failed():
    run = run_case(CASE_A_DIR, FailingExtractor({"A-batch-001"}), max_chars=20_000)

    assert run.status == "failed"
    assert run.events == []
    assert run.batches[0].attempts == 3


def test_unexpected_errors_are_not_hidden():
    class BrokenExtractor:
        info = FakeExtractor.info

        def extract(self, batch: Batch) -> ExtractionResult:
            raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        run_case(CASE_A_DIR, BrokenExtractor(), max_chars=20_000)


def test_the_same_event_found_in_two_batches_is_kept_twice():
    class SameEventExtractor:
        info = FakeExtractor.info

        def extract(self, batch: Batch) -> ExtractionResult:
            quote = make_quote(1, "# Birch Clinic: initial assessment", document_id="B01")
            return ExtractionResult(events=[make_event(evidence=[quote])], attempts=1)

    run = run_case(CASE_B_DIR, SameEventExtractor(), max_chars=900)

    assert [event.event_id for event in run.events] == ["B-batch-001-e01", "B-batch-002-e01", "B-batch-003-e01"]
    # Only the first batch contains B01, so the other two quotes fail the citation check.
    assert [len(event.citation_errors) for event in run.events] == [0, 1, 1]


def test_extraction_code_never_imports_the_evaluator():
    for path in (REPO_ROOT / "src" / "evidence_timeline").glob("*.py"):
        if path.name not in {"evaluation.py", "cli.py"}:
            assert "evidence_timeline.evaluation" not in path.read_text(encoding="utf-8"), path.name
