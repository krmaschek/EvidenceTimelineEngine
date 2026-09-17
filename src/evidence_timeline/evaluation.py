"""Evaluator only: score a saved run against the dataset's reference events.

The extraction code never imports this module, so reference answers cannot
reach a prompt.

A person decides which prediction matches which reference event and writes
that into a review file. This module checks the review against the dataset's
matching rules and counts the results. Nothing is matched automatically.
"""

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from evidence_timeline.models import EventStatus, EventType, TimelineEvent, TimelineRun

NoMatchReason = Literal["duplicate", "wrong_date", "wrong_status", "wrong_type", "not_a_reference_event", "other"]
Dates = tuple[dt.date | None, tuple[dt.date, dt.date] | None, list[dt.date]]  # exact date, range, alternatives


# --- Reference answers (the dataset's gold files) --------------------------------


class DateRange(BaseModel):
    start: dt.date
    end: dt.date


class ReferenceEvent(BaseModel):
    """The fields of a gold event that scoring needs. Other fields in the file are ignored."""

    event_id: str
    event_type: EventType
    status: EventStatus
    description: str
    date: dt.date | None
    date_interval: DateRange | None
    alternative_dates: list[dt.date]


class ReferenceCase(BaseModel):
    case_id: str
    events: list[ReferenceEvent]


# --- The human review ------------------------------------------------------------


class Decision(BaseModel):
    prediction_id: str
    prediction: str = ""  # summary for the reviewer, written by review-template
    match: str | None = None  # ID of the matching reference event
    reason: NoMatchReason | None = None  # why the prediction matches nothing
    note: str = ""


class Review(BaseModel):
    case_id: str
    run_id: str
    reviewer: str = ""
    reference_events: list[str] = []  # summaries for the reviewer, written by review-template
    decisions: list[Decision]


# --- Results ---------------------------------------------------------------------


class Score(BaseModel):
    predictions: int
    references: int
    matched: int

    @property
    def precision(self) -> float | None:
        return self.matched / self.predictions if self.predictions else None

    @property
    def recall(self) -> float | None:
        return self.matched / self.references if self.references else None


class CaseResult(BaseModel):
    case_id: str
    run_status: str
    fake_extractor: bool
    all_events: Score
    completed_events: Score
    no_match_reasons: dict[str, int]
    missed_references: list[str]
    quotes: int
    failed_citations: int


# --- Functions -------------------------------------------------------------------


def load_run(path: Path) -> TimelineRun:
    return TimelineRun.model_validate_json(path.read_text(encoding="utf-8"))


def load_review(path: Path) -> Review:
    return Review.model_validate_json(path.read_text(encoding="utf-8"))


def load_reference(gold_dir: Path, case_id: str) -> ReferenceCase:
    return ReferenceCase.model_validate_json((gold_dir / f"case_{case_id}.json").read_text(encoding="utf-8"))


def prediction_dates(event: TimelineEvent) -> Dates:
    date_range = None
    if event.date_earliest is not None and event.date_latest is not None:
        date_range = (event.date_earliest, event.date_latest)
    return event.date, date_range, sorted(set(event.alternative_dates))


def reference_dates(event: ReferenceEvent) -> Dates:
    date_range = (event.date_interval.start, event.date_interval.end) if event.date_interval else None
    return event.date, date_range, sorted(set(event.alternative_dates))


def describe_dates(dates: Dates) -> str:
    date, date_range, alternatives = dates
    if date:
        return str(date)
    if date_range:
        return f"between {date_range[0]} and {date_range[1]}"
    if alternatives:
        return "conflicting: " + " / ".join(map(str, alternatives))
    return "no date"


def create_review_template(run: TimelineRun, reference: ReferenceCase) -> Review:
    """A review with every decision left empty. The reviewer fills in each one."""
    return Review(
        case_id=run.case_id,
        run_id=run.run_id,
        reference_events=[
            f"{e.event_id} | {e.event_type} | {e.status} | {describe_dates(reference_dates(e))} | {e.description}"
            for e in reference.events
        ],
        decisions=[
            Decision(
                prediction_id=e.event_id,
                prediction=f"{e.event_type} | {e.status} | {describe_dates(prediction_dates(e))} | {e.description}",
            )
            for e in run.events
        ],
    )


def rule_problems(prediction: TimelineEvent, reference: ReferenceEvent) -> list[str]:
    """The dataset's matching rules that can be checked without judging wording."""
    problems: list[str] = []
    if prediction.event_type != reference.event_type:
        problems.append(f"event type: expected {reference.event_type}, got {prediction.event_type}")
    if prediction.status != reference.status:
        problems.append(f"status: expected {reference.status}, got {prediction.status}")
    if prediction_dates(prediction) != reference_dates(reference):
        expected = describe_dates(reference_dates(reference))
        predicted = describe_dates(prediction_dates(prediction))
        problems.append(f"date: expected {expected}, got {predicted}")
    return problems


def find_review_problems(run: TimelineRun, reference: ReferenceCase, review: Review) -> list[str]:
    predictions = {event.event_id: event for event in run.events}
    references = {event.event_id: event for event in reference.events}
    problems: list[str] = []

    if review.run_id != run.run_id:
        problems.append(f"the review is for run {review.run_id}, not for run {run.run_id}")
    if sorted(d.prediction_id for d in review.decisions) != sorted(predictions):
        problems.append("the review needs exactly one decision for every prediction in the run")

    match_counts = Counter(d.match for d in review.decisions)
    for decision in review.decisions:
        name = decision.prediction_id
        if (decision.match is None) == (decision.reason is None):
            problems.append(f"{name}: fill in either 'match' or 'reason'")
        elif decision.match is not None:
            if decision.match not in references:
                problems.append(f"{name}: there is no reference event {decision.match}")
            elif match_counts[decision.match] > 1:
                problems.append(f"{name}: {decision.match} is matched more than once; mark the extras as duplicate")
            elif name in predictions:
                for problem in rule_problems(predictions[name], references[decision.match]):
                    problems.append(f"{name} -> {decision.match}: {problem}")
    return problems


def evaluate_case(run: TimelineRun, reference: ReferenceCase, review: Review) -> CaseResult:
    problems = find_review_problems(run, reference, review)
    if problems:
        raise ValueError("The review cannot be scored:\n- " + "\n- ".join(problems))

    matched = {d.match for d in review.decisions if d.match is not None}
    completed = {e.event_id for e in reference.events if e.status == "completed"}
    return CaseResult(
        case_id=run.case_id,
        run_status=run.status,
        fake_extractor=run.extractor.kind == "fake",
        all_events=Score(predictions=len(run.events), references=len(reference.events), matched=len(matched)),
        # A match requires equal status, so completed references are only matched by completed predictions.
        completed_events=Score(
            predictions=sum(e.status == "completed" for e in run.events),
            references=len(completed),
            matched=len(matched & completed),
        ),
        no_match_reasons=dict(Counter(d.reason for d in review.decisions if d.reason is not None)),
        missed_references=[f"{e.event_id} ({e.description})" for e in reference.events if e.event_id not in matched],
        quotes=sum(len(e.evidence) for e in run.events),
        failed_citations=sum(len(e.citation_errors) for e in run.events),
    )


def format_results(results: list[CaseResult]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"Case {result.case_id} (run status: {result.run_status})")
        if result.fake_extractor:
            lines.append("  PIPELINE TEST ONLY: fake extractor output says nothing about extraction quality")
        if result.run_status != "completed":
            lines.append("  WARNING: some batches failed, so some misses are not the model's fault")
        lines += [
            f"  all events:       {format_score(result.all_events)}",
            f"  completed events: {format_score(result.completed_events)}",
            f"  no-match reasons: {result.no_match_reasons or 'none'}",
            f"  missed references: {', '.join(result.missed_references) or 'none'}",
            f"  citation check (not part of the scores): {result.failed_citations} of {result.quotes} quotes failed",
            "",
        ]

    all_events = add_scores([result.all_events for result in results])
    completed_events = add_scores([result.completed_events for result in results])
    lines += [
        "All cases together",
        f"  all events:       {format_score(all_events)}",
        f"  completed events: {format_score(completed_events)}",
    ]
    return "\n".join(lines)


def add_scores(scores: list[Score]) -> Score:
    return Score(
        predictions=sum(score.predictions for score in scores),
        references=sum(score.references for score in scores),
        matched=sum(score.matched for score in scores),
    )


def format_score(score: Score) -> str:
    return (
        f"{score.matched} matched, {score.predictions - score.matched} false positives, "
        f"{score.references - score.matched} missed, "
        f"precision {percent(score.precision)}, recall {percent(score.recall)}"
    )


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"
