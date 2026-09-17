"""Turn extracted events into timeline events and sort them.

Nothing is merged here: events on the same day stay separate, and the same event
found in two batches appears twice. Deduplication comes in a later phase.
"""

import datetime as dt

from evidence_timeline.citations import check_citation
from evidence_timeline.models import Batch, ExtractedEvent, TimelineEvent

ORDERING_NOTE = (
    "Events are sorted by their earliest possible date, then their latest possible date. "
    "Events without a date come last. Events with the same dates keep source order "
    "(document, line); that order says nothing about the time of day."
)

DATE_REVIEW_REASONS = {
    "approximate": "approximate date",
    "conflicting": "conflicting dates",
    "unknown": "no date",
}


def build_timeline_events(batch: Batch, extracted_events: list[ExtractedEvent]) -> list[TimelineEvent]:
    timeline_events = []
    for number, event in enumerate(extracted_events, start=1):
        citation_errors = []
        for quote in event.evidence:
            error = check_citation(quote, batch)
            if error:
                citation_errors.append(error)

        # Our own reasons come first, so flagging never depends on the model. Then the model's reasons.
        reasons: list[str] = []
        if event.date_precision in DATE_REVIEW_REASONS:
            reasons.append(DATE_REVIEW_REASONS[event.date_precision])
        reasons += [f"citation check failed: {error}" for error in citation_errors]
        reasons += [reason for reason in event.review_reasons if reason not in reasons]

        timeline_events.append(
            TimelineEvent(
                **event.model_dump(exclude={"review_reasons"}),
                review_reasons=reasons,
                event_id=f"{batch.batch_id}-e{number:02d}",
                batch_id=batch.batch_id,
                citation_errors=citation_errors,
                needs_review=bool(reasons),
            )
        )
    return timeline_events


def date_range(event: ExtractedEvent) -> tuple[dt.date, dt.date] | None:
    """The earliest and latest day the event can have happened on."""
    if event.date is not None:
        return event.date, event.date
    if event.date_earliest is not None and event.date_latest is not None:
        return event.date_earliest, event.date_latest
    if event.alternative_dates:
        return min(event.alternative_dates), max(event.alternative_dates)
    return None


def sort_key(event: TimelineEvent) -> tuple[object, ...]:
    dates = date_range(event)
    first_quote = event.evidence[0]
    return (
        dates is None,  # False sorts before True, so undated events go last
        dates or (dt.date.min, dt.date.min),
        first_quote.document_id,
        first_quote.line_start,
        event.event_id,
    )


def sort_timeline(events: list[TimelineEvent]) -> list[TimelineEvent]:
    return sorted(events, key=sort_key)
