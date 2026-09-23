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

from evidence_timeline.models import EventStatus, TimelineEvent, TimelineRun

NoMatchReason = Literal["duplicate", "wrong_date", "wrong_status", "not_a_reference_event", "other"]
Dates = tuple[dt.date | None, tuple[dt.date, dt.date] | None, list[dt.date]]  # exact date, range, alternatives


# --- Reference answers (the dataset's gold files) --------------------------------


class DateRange(BaseModel):
    start: dt.date
    end: dt.date


class ReferenceSource(BaseModel):
    """Where a gold event is written. Shown to the reviewer; not scored."""

    document_id: str
    line_start: int


class ReferenceEvent(BaseModel):
    """The fields of a gold event that scoring needs. Other fields in the file are ignored."""

    event_id: str
    event_type: str
    status: EventStatus
    description: str
    date: dt.date | None
    date_interval: DateRange | None
    alternative_dates: list[dt.date]
    date_scored: bool = True  # False when the gold wording is too vague to turn into a date
    sources: list[ReferenceSource]


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
    type_right: int  # matched events that also have the reference's event type
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


def describe_reference_date(event: ReferenceEvent) -> str:
    if not event.date_scored:
        return "date not scored"
    return describe_dates(reference_dates(event))


def create_review_template(run: TimelineRun, reference: ReferenceCase) -> Review:
    """A review with every decision left empty. The reviewer fills in each one.

    Each summary starts with where the event is written, e.g. "A02:8", because a
    prediction and its gold event are usually on the same line.
    """
    reference_events = []
    for event in reference.events:
        source = event.sources[0]
        date = describe_reference_date(event)
        reference_events.append(
            f"{event.event_id} | {source.document_id}:{source.line_start} | {event.event_type} | {event.status} | "
            f"{date} | {event.description}"
        )

    decisions = []
    for event in run.events:
        quote = event.evidence[0]
        date = describe_dates(prediction_dates(event))
        summary = f"{quote.document_id}:{quote.line_start} | {event.event_type} | {event.status} | {date} | {event.description}"
        decisions.append(Decision(prediction_id=event.event_id, prediction=summary))

    return Review(case_id=run.case_id, run_id=run.run_id, reference_events=reference_events, decisions=decisions)


def rule_problems(prediction: TimelineEvent, reference: ReferenceEvent) -> list[str]:
    """The dataset's matching rules that can be checked without judging wording.

    The event type is not one of them. It is scored on its own, so a found event
    with the wrong type is not also counted as a false positive and a miss.
    """
    problems: list[str] = []
    if prediction.status != reference.status:
        problems.append(f"status: expected {reference.status}, got {prediction.status}")
    # An unscored gold date is not compared, so any predicted date can match it.
    if reference.date_scored and prediction_dates(prediction) != reference_dates(reference):
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

    predictions = {event.event_id: event for event in run.events}
    references = {event.event_id: event for event in reference.events}
    matches = [d for d in review.decisions if d.match is not None]
    type_right = sum(predictions[d.prediction_id].event_type == references[d.match].event_type for d in matches)

    matched = {d.match for d in matches}
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
        type_right=type_right,
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
            f"  event type right: {format_type_right(result.type_right, result.all_events.matched)}",
            f"  no-match reasons: {result.no_match_reasons or 'none'}",
            f"  missed references: {', '.join(result.missed_references) or 'none'}",
            f"  citation check (not part of the scores): {result.failed_citations} of {result.quotes} quotes failed",
            "",
        ]

    all_events = add_scores([result.all_events for result in results])
    completed_events = add_scores([result.completed_events for result in results])
    type_right = sum(result.type_right for result in results)
    lines += [
        "All cases together",
        f"  all events:       {format_score(all_events)}",
        f"  completed events: {format_score(completed_events)}",
        f"  event type right: {format_type_right(type_right, all_events.matched)}",
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


def format_type_right(type_right: int, matched: int) -> str:
    share = type_right / matched if matched else None
    return f"{type_right} of {matched} matched events ({percent(share)})"


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"
