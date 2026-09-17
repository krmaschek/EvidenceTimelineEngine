import pytest
from helpers import make_batch, make_event, make_quote

from evidence_timeline.citations import check_citation
from evidence_timeline.models import EvidenceQuote
from evidence_timeline.timeline import build_timeline_events

VISIT = "Patient attended a visit on 6 January 2025."
BATCH = make_batch(["# Title", "", VISIT, "The second sentence", "continues here."])


def test_exact_quote_at_the_cited_line_passes():
    assert check_citation(make_quote(3, VISIT, date_text="6 January 2025"), BATCH) is None


def test_quote_over_two_lines_passes():
    quote = EvidenceQuote(
        document_id="D01", line_start=4, line_end=5, quote="The second sentence\ncontinues here.", date_text=None
    )

    assert check_citation(quote, BATCH) is None


@pytest.mark.parametrize(
    ("quote", "error"),
    [
        (make_quote(3, VISIT, document_id="D99"), "D99 lines 3-3 are not in this batch"),
        (make_quote(9, VISIT), "D01 lines 9-9 are not in this batch"),
        (make_quote(4, VISIT), "quote not found in D01 lines 4-4"),
        (make_quote(3, "Patient visited on 6 January 2025."), "quote not found in D01 lines 3-3"),
        (make_quote(3, VISIT, date_text="7 January 2025"), "date_text '7 January 2025' is not part of the quote"),
    ],
)
def test_wrong_citations_are_reported(quote, error):
    assert check_citation(quote, BATCH) == error


def test_a_correct_location_does_not_prove_the_event():
    # The quote is about a completed visit, but the event claims a planned procedure.
    # The citation check still passes: judging whether a quote supports an event is left to a person.
    event = make_event(event_type="procedure", status="planned", evidence=[make_quote(3, VISIT)])

    [timeline_event] = build_timeline_events(BATCH, [event])

    assert timeline_event.citation_errors == []
