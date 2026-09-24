"""Merging is pure: the tests hand it the matcher's answers, so no model is involved."""

import datetime as dt

from helpers import APPROXIMATE_MARCH, NO_DATE, make_quote, make_timeline_event

from evidence_timeline.merge import candidates, merge_events
from evidence_timeline.models import PairDecision

X_RAY = {"event_type": "procedure", "description": "Right ankle X-ray"}


def pairs_of(events):
    return [(pair.a.event_id, pair.b.event_id) for pair in candidates(events)]


def same(first, second):
    return PairDecision(a=first.event_id, b=second.event_id, same_event=True, reason="test")


def different(first, second):
    return PairDecision(a=first.event_id, b=second.event_id, same_event=False, reason="test")


def test_events_of_a_different_kind_are_never_asked_about():
    visit = make_timeline_event("visit")
    x_ray = make_timeline_event("x-ray", **X_RAY)

    assert pairs_of([visit, x_ray]) == []


def test_a_planned_and_a_completed_event_stay_separate():
    done = make_timeline_event("done", **X_RAY)
    planned = make_timeline_event("planned", status="planned", **X_RAY)

    assert pairs_of([done, planned]) == []


def test_events_close_in_time_are_asked_about_and_distant_ones_are_not():
    first = make_timeline_event("6 Jan", date=dt.date(2025, 1, 6))
    two_days_later = make_timeline_event("8 Jan", date=dt.date(2025, 1, 8))
    three_weeks_later = make_timeline_event("27 Jan", date=dt.date(2025, 1, 27))

    assert pairs_of([first, two_days_later, three_weeks_later]) == [("6 Jan", "8 Jan")]


def test_an_event_without_a_date_is_never_ruled_out_by_its_date():
    dated = make_timeline_event("dated", date=dt.date(2025, 1, 6))
    undated = make_timeline_event("undated", **NO_DATE)

    assert pairs_of([dated, undated]) == [("dated", "undated")]


def test_events_nobody_linked_come_back_unchanged():
    events = [make_timeline_event("one"), make_timeline_event("two", **X_RAY)]

    assert merge_events(events, []) == events


def test_a_confirmed_pair_becomes_one_event_that_keeps_both_quotes():
    first = make_timeline_event("first", evidence=[make_quote(9, "X-ray performed", "B01")], **X_RAY)
    second = make_timeline_event("second", evidence=[make_quote(8, "the prior X-ray", "B02")], **X_RAY)

    merged = merge_events([first, second], [same(first, second)])

    assert len(merged) == 1
    assert [quote.document_id for quote in merged[0].evidence] == ["B01", "B02"]
    assert merged[0].merged_from == ["first", "second"]
    assert merged[0].description == "Right ankle X-ray"
    assert merged[0].date == dt.date(2025, 1, 6)


def test_the_description_comes_from_the_document_that_records_it_first():
    later = make_timeline_event("later", description="The prior X-ray", evidence=[make_quote(9, "q", "B03")])
    first = make_timeline_event("first", description="An X-ray was performed", evidence=[make_quote(9, "q", "B01")])

    merged = merge_events([later, first], [same(later, first)])

    assert merged[0].description == "An X-ray was performed"


def test_a_link_is_followed_through_to_events_nobody_was_asked_about():
    a = make_timeline_event("a", evidence=[make_quote(1, "q", "B01")])
    b = make_timeline_event("b", evidence=[make_quote(1, "q", "B02")])
    c = make_timeline_event("c", evidence=[make_quote(1, "q", "B03")])

    merged = merge_events([a, b, c], [same(a, b), same(b, c)])

    assert len(merged) == 1
    assert merged[0].merged_from == ["a", "b", "c"]


def test_a_link_is_not_followed_to_an_event_said_to_be_different():
    # b mentions both a and c, e.g. "notified by email, and answered thirteen minutes later".
    a = make_timeline_event("a", evidence=[make_quote(1, "q", "B01")])
    b = make_timeline_event("b", evidence=[make_quote(1, "q", "B02")])
    c = make_timeline_event("c", evidence=[make_quote(1, "q", "B03")])

    merged = merge_events([a, b, c], [same(a, b), same(b, c), different(a, c)])

    assert [event.merged_from for event in merged] == [["a", "b"], []]
    assert merged[1].event_id == "c"


def test_sources_that_disagree_about_the_day_become_one_conflicting_event():
    fourteenth = make_timeline_event("14th", date=dt.date(2025, 4, 14), evidence=[make_quote(8, "q", "C03")])
    fifteenth = make_timeline_event("15th", date=dt.date(2025, 4, 15), evidence=[make_quote(9, "q", "C04")])

    merged = merge_events([fourteenth, fifteenth], [same(fourteenth, fifteenth)])

    assert merged[0].date is None
    assert merged[0].date_precision == "conflicting"
    assert merged[0].alternative_dates == [dt.date(2025, 4, 14), dt.date(2025, 4, 15)]
    assert merged[0].needs_review
    assert "conflicting dates" in merged[0].review_reasons


def test_a_stated_day_is_kept_over_a_source_that_only_knows_the_month():
    vague = make_timeline_event("vague", evidence=[make_quote(1, "q", "B01")], **APPROXIMATE_MARCH)
    exact = make_timeline_event("exact", date=dt.date(2025, 3, 4), evidence=[make_quote(1, "q", "B02")])

    merged = merge_events([vague, exact], [same(vague, exact)])

    assert merged[0].date == dt.date(2025, 3, 4)
    assert merged[0].date_precision == "exact"
    assert merged[0].review_reasons == []  # the "approximate date" reason no longer applies
