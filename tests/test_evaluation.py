"""Scoring tests on a small made-up reference case (not the dataset's answer keys)."""

import datetime as dt
import re

import pytest
from helpers import APPROXIMATE_MARCH, GOLD_DIR, NO_DATE, make_run, make_timeline_event

from evidence_timeline.evaluation import (
    Decision,
    ReferenceCase,
    Review,
    Score,
    create_review_template,
    evaluate_case,
    format_results,
    load_reference,
)
from evidence_timeline.models import TimelineRun


def reference_event(event_id, event_type, status, date="2025-01-06", date_interval=None):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "status": status,
        "description": event_id,
        "date": date,
        "date_interval": date_interval,
        "alternative_dates": [],
    }


REFERENCE = ReferenceCase.model_validate(
    {
        "case_id": "R",
        "events": [
            reference_event("R-E01", "visit", "completed"),
            reference_event("R-E02", "procedure", "completed"),
            reference_event("R-E03", "procedure", "planned", date="2025-02-01"),
            reference_event("R-E04", "visit", "completed", None, {"start": "2025-03-01", "end": "2025-03-31"}),
        ],
    }
)


def predictions() -> TimelineRun:
    return make_run(
        [
            make_timeline_event("p1"),  # R-E01
            make_timeline_event("p2", event_type="procedure"),  # R-E02
            make_timeline_event("p3", event_type="procedure"),  # R-E02 again
            make_timeline_event("p4", event_type="procedure", date=dt.date(2025, 2, 1)),  # R-E03, but not planned
            make_timeline_event("p5", **APPROXIMATE_MARCH),  # R-E04
            make_timeline_event("p6", **NO_DATE),  # not a reference event
        ]
    )


def review(run: TimelineRun, **changes: dict) -> Review:
    decisions = {
        "p1": {"match": "R-E01"},
        "p2": {"match": "R-E02"},
        "p3": {"reason": "duplicate"},
        "p4": {"reason": "wrong_status"},
        "p5": {"match": "R-E04"},
        "p6": {"reason": "not_a_reference_event"},
    }
    decisions.update(changes)
    return Review(
        case_id="R",
        run_id=run.run_id,
        decisions=[Decision(prediction_id=prediction_id, **fields) for prediction_id, fields in decisions.items()],
    )


def test_scores_follow_one_to_one_matching():
    run = predictions()

    result = evaluate_case(run, REFERENCE, review(run))

    assert result.all_events == Score(predictions=6, references=4, matched=3)
    assert (result.all_events.precision, result.all_events.recall) == (0.5, 0.75)
    # The planned R-E03 is left out of the completed-events view.
    assert result.completed_events == Score(predictions=6, references=3, matched=3)
    assert result.no_match_reasons == {"duplicate": 1, "wrong_status": 1, "not_a_reference_event": 1}
    assert result.missed_references == ["R-E03 (R-E03)"]
    assert (result.failed_citations, result.quotes) == (0, 6)


def test_precision_is_not_available_without_predictions():
    run = make_run([])

    result = evaluate_case(run, REFERENCE, Review(case_id="R", run_id=run.run_id, decisions=[]))

    assert result.all_events.precision is None
    assert result.all_events.recall == 0
    assert "precision N/A" in format_results([result])


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"p3": {"match": "R-E02"}}, "p3: R-E02 is matched more than once"),
        ({"p3": {}}, "p3: fill in either 'match' or 'reason'"),
        ({"p3": {"match": "R-E02", "reason": "duplicate"}}, "p3: fill in either 'match' or 'reason'"),
        ({"p1": {"match": "R-E99"}}, "p1: there is no reference event R-E99"),
        ({"p4": {"match": "R-E03"}}, "p4 -> R-E03: status: expected planned, got completed"),
        (
            {"p1": {"reason": "other"}, "p2": {"match": "R-E01"}},
            "p2 -> R-E01: event type: expected visit, got procedure",
        ),
        (
            {"p5": {"reason": "other"}, "p6": {"match": "R-E04"}},
            "p6 -> R-E04: date: expected between 2025-03-01 and 2025-03-31, got no date",
        ),
    ],
)
def test_reviews_that_break_the_rules_are_rejected(changes, problem):
    run = predictions()

    with pytest.raises(ValueError, match=re.escape(problem)):
        evaluate_case(run, REFERENCE, review(run, **changes))


def test_an_invented_day_cannot_match_an_approximate_reference():
    run = make_run([make_timeline_event("p5", date=dt.date(2025, 3, 15))])
    decisions = [Decision(prediction_id="p5", match="R-E04")]

    with pytest.raises(ValueError, match="expected between 2025-03-01 and 2025-03-31, got 2025-03-15"):
        evaluate_case(run, REFERENCE, Review(case_id="R", run_id=run.run_id, decisions=decisions))


def test_review_must_belong_to_the_run_and_cover_every_prediction():
    run = predictions()
    wrong_review = review(run)
    wrong_review.run_id = "another-run"
    wrong_review.decisions = wrong_review.decisions[1:]

    with pytest.raises(ValueError) as error:
        evaluate_case(run, REFERENCE, wrong_review)

    assert "the review is for run another-run" in str(error.value)
    assert "one decision for every prediction" in str(error.value)


def test_review_template_leaves_every_decision_empty():
    run = predictions()

    template = create_review_template(run, REFERENCE)

    assert [decision.prediction_id for decision in template.decisions] == ["p1", "p2", "p3", "p4", "p5", "p6"]
    assert all(decision.match is None and decision.reason is None for decision in template.decisions)
    assert template.decisions[4].prediction == "visit | completed | between 2025-03-01 and 2025-03-31 | Visit"
    assert template.reference_events[3] == "R-E04 | visit | completed | between 2025-03-01 and 2025-03-31 | R-E04"
    with pytest.raises(ValueError):
        evaluate_case(run, REFERENCE, template)


def test_report_marks_fake_and_partial_runs():
    run = make_run([], kind="fake", status="partial")

    result = evaluate_case(run, REFERENCE, Review(case_id="R", run_id=run.run_id, decisions=[]))

    report = format_results([result])
    assert "PIPELINE TEST ONLY" in report
    assert "WARNING: some batches failed" in report


def test_totals_add_up_all_cases():
    run = predictions()
    result = evaluate_case(run, REFERENCE, review(run))

    report = format_results([result, result])

    assert "all events:       6 matched, 6 false positives, 2 missed, precision 50.0%, recall 75.0%" in report


def test_answer_keys_of_the_development_cases_load():
    assert len(load_reference(GOLD_DIR, "A").events) == 8
    assert len(load_reference(GOLD_DIR, "B").events) == 8
