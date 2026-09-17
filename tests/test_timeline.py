import datetime as dt

from helpers import (
    APPROXIMATE_MARCH,
    CONFLICTING_MARCH,
    NO_DATE,
    make_batch,
    make_event,
    make_quote,
    make_timeline_event,
)

from evidence_timeline.timeline import build_timeline_events, sort_timeline


def ids(events):
    return [event.event_id for event in events]


def test_same_day_events_stay_separate_in_source_order():
    visit = make_timeline_event("visit", evidence=[make_quote(8)])
    x_ray = make_timeline_event("x-ray", event_type="procedure", evidence=[make_quote(9)])

    assert ids(sort_timeline([x_ray, visit])) == ["visit", "x-ray"]


def test_uncertain_and_missing_dates_are_sorted_without_inventing_a_day():
    events = [
        make_timeline_event("no date", **NO_DATE),
        make_timeline_event("15 March", date=dt.date(2025, 3, 15)),
        make_timeline_event("10 or 12 March", **CONFLICTING_MARCH),
        make_timeline_event("sometime in March", **APPROXIMATE_MARCH),
        make_timeline_event("1 March", date=dt.date(2025, 3, 1)),
        make_timeline_event("February", date=dt.date(2025, 2, 1)),
    ]

    ordered = sort_timeline(events)

    assert ids(ordered) == ["February", "1 March", "sometime in March", "10 or 12 March", "15 March", "no date"]
    assert ordered[2].date is None


def test_sorting_does_not_depend_on_input_order():
    events = [
        make_timeline_event(f"e{n}", date=dt.date(2025, 1, 1 + n % 3), evidence=[make_quote(n % 4 + 1)])
        for n in range(8)
    ]

    expected = ids(sort_timeline(events))

    assert ids(sort_timeline(list(reversed(events)))) == expected
    assert ids(sort_timeline(events[3:] + events[:3])) == expected


def test_review_reasons_come_from_dates_citations_and_the_model():
    batch = make_batch(["real text"])
    events = [
        make_event(evidence=[make_quote(1, "real text")]),
        make_event(**APPROXIMATE_MARCH, evidence=[make_quote(1, "real text")], review_reasons=["unclear clinic"]),
        make_event(evidence=[make_quote(1, "invented text")]),
    ]

    timeline_events = build_timeline_events(batch, events)

    assert [event.review_reasons for event in timeline_events] == [
        [],
        ["approximate date", "unclear clinic"],
        ["citation check failed: quote not found in D01 lines 1-1"],
    ]
    assert [event.needs_review for event in timeline_events] == [False, True, True]
    assert ids(timeline_events) == ["X-batch-001-e01", "X-batch-001-e02", "X-batch-001-e03"]


def test_status_and_date_wording_are_kept():
    line = "An MRI is booked for 2 May 2025."
    event = make_event(
        event_type="procedure",
        status="planned",
        date=dt.date(2025, 5, 2),
        evidence=[make_quote(1, line, date_text="2 May 2025")],
    )

    [timeline_event] = build_timeline_events(make_batch([line]), [event])

    assert timeline_event.status == "planned"
    assert timeline_event.date == dt.date(2025, 5, 2)
    assert timeline_event.evidence[0].date_text == "2 May 2025"
    assert timeline_event.citation_errors == []
