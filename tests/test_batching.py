import pytest
from helpers import CASE_A_DIR, CASE_B_DIR

from evidence_timeline.batching import build_batches, check_coverage
from evidence_timeline.documents import load_documents
from evidence_timeline.models import Batch, DocumentSpan, SourceDocument


def test_documents_keep_their_ids_and_line_numbers():
    documents = load_documents(CASE_A_DIR)

    assert [document.document_id for document in documents] == ["A01", "A02", "A03", "A04"]
    assert documents[0].lines[0] == "# Harbor Clinic: initial visit"
    assert documents[0].lines[1] == ""  # empty lines are kept, so numbering matches the file
    assert documents[0].lines[9 - 1] == "A left wrist X-ray was performed at Harbor Clinic on 6 January 2025."


def test_small_case_fits_in_one_batch():
    documents = load_documents(CASE_B_DIR)

    batches = build_batches("B", documents, max_chars=20_000)

    assert len(batches) == 1
    check_coverage(documents, batches)


def test_smaller_budget_gives_more_batches_without_losing_lines():
    documents = load_documents(CASE_B_DIR)

    batches = build_batches("B", documents, max_chars=900)

    documents_per_batch = [[span.document_id for span in batch.spans] for batch in batches]
    assert documents_per_batch == [["B01", "B02"], ["B03", "B04"], ["B05"]]
    assert [batch.batch_id for batch in batches] == ["B-batch-001", "B-batch-002", "B-batch-003"]
    check_coverage(documents, batches)


def test_long_document_is_cut_between_lines():
    document = SourceDocument(document_id="D01", sha256="", lines=[f"line {n}" for n in range(1, 11)])

    batches = build_batches("X", [document], max_chars=20)

    spans = [span for batch in batches for span in batch.spans]
    assert [(span.first_line, span.last_line) for span in spans] == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    check_coverage([document], batches)


def test_line_longer_than_the_budget_gets_its_own_batch():
    document = SourceDocument(document_id="D01", sha256="", lines=["short", "x" * 100, "short"])

    batches = build_batches("X", [document], max_chars=20)

    assert [batch.spans[0].lines for batch in batches] == [["short"], ["x" * 100], ["short"]]


def test_coverage_check_catches_missing_and_repeated_lines():
    document = SourceDocument(document_id="D01", sha256="", lines=["a", "b"])
    first_line_only = Batch(batch_id="X-1", spans=[DocumentSpan(document_id="D01", first_line=1, lines=["a"])])
    both_lines = Batch(batch_id="X-2", spans=[DocumentSpan(document_id="D01", first_line=1, lines=["a", "b"])])

    with pytest.raises(RuntimeError):
        check_coverage([document], [first_line_only])
    with pytest.raises(RuntimeError):
        check_coverage([document], [first_line_only, both_lines])
