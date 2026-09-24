"""Merge events that several documents describe.

Each batch is extracted on its own, so an event mentioned in three documents comes
back three times. `candidates` picks the pairs worth checking, and `merge_events`
applies the answers. The answering (usually by an LLM) happens outside, so this
module stays pure and easy to test.
"""

import datetime as dt
import itertools
from typing import Protocol

from evidence_timeline.models import EventPair, MatchResult, PairDecision, TimelineEvent
from evidence_timeline.timeline import DATE_REVIEW_REASONS, date_range

# How many days apart two records of the same event may be. Wide enough for sources that
# are off by a day or two, narrow enough to keep repeated events (like weekly visits) apart.
WINDOW = dt.timedelta(days=3)


def candidates(events: list[TimelineEvent]) -> list[EventPair]:
    """Pairs that might be the same event.

    A pair is kept unless a rule clearly rules it out: an extra question is cheap,
    but a skipped pair can never be merged.
    """
    pairs = []
    for first, second in itertools.combinations(events, 2):
        if first.event_type != second.event_type:
            continue
        if first.status != second.status:
            continue  # planned and done are different events
        if dates_are_far_apart(first, second):
            continue
        pairs.append(EventPair(a=first, b=second))
    return pairs


def dates_are_far_apart(first: TimelineEvent, second: TimelineEvent) -> bool:
    """True when the two events cannot be the same because of their dates.

    An event with no date could have happened on any day, so it never rules a pair out.
    """
    first_days, second_days = date_range(first), date_range(second)
    if first_days is None or second_days is None:
        return False
    return first_days[0] - WINDOW > second_days[1] or second_days[0] - WINDOW > first_days[1]


def merge_events(events: list[TimelineEvent], decisions: list[PairDecision]) -> list[TimelineEvent]:
    """Apply the matcher's answers. Events that were not linked come back unchanged."""
    return [merge_group(group) for group in group_events(events, decisions)]


def group_events(events: list[TimelineEvent], decisions: list[PairDecision]) -> list[list[TimelineEvent]]:
    """Put events that a "same" answer links into the same group.

    Linking is followed through: if A goes with B and B goes with C, all three end up
    together, unless the matcher said A and C are different. Without that check, one record
    that mentions two events ("notified by email, and answered thirteen minutes later")
    would chain them into one.
    """
    group_of = {event.event_id: number for number, event in enumerate(events)}
    different = [(decision.a, decision.b) for decision in decisions if not decision.same_event]
    for decision in decisions:
        if not decision.same_event:
            continue
        keep, replace = sorted((group_of[decision.a], group_of[decision.b]))
        if any({group_of[a], group_of[b]} == {keep, replace} for a, b in different):
            continue  # a member of one group was said to differ from a member of the other
        for event_id, number in group_of.items():
            if number == replace:
                group_of[event_id] = keep

    groups: dict[int, list[TimelineEvent]] = {}
    for event in events:  # groups keep the order of their first event
        groups.setdefault(group_of[event.event_id], []).append(event)
    return list(groups.values())


def merge_group(group: list[TimelineEvent]) -> TimelineEvent:
    """Turn one group into one event. Every quote is kept; only the wording is chosen."""
    if len(group) == 1:
        return group[0]

    members = sorted(group, key=first_source)
    fields = members[0].model_dump()  # the earliest mention supplies the description
    fields |= merged_dates(members)
    fields |= {
        "evidence": sorted(
            [quote for member in members for quote in member.evidence],
            key=lambda quote: (quote.document_id, quote.line_start),
        ),
        "citation_errors": [error for member in members for error in member.citation_errors],
        "merged_from": [member.event_id for member in members],
    }
    reasons = merged_reasons(members, str(fields["date_precision"]))
    fields |= {"review_reasons": reasons, "needs_review": bool(reasons)}
    return TimelineEvent(**fields)  # rebuilt, so the date rules are checked again


def first_source(event: TimelineEvent) -> tuple[str, int, str]:
    """Where the event is first mentioned.

    That is usually the original record; later mentions refer back to it.
    """
    quote = event.evidence[0]
    return quote.document_id, quote.line_start, event.event_id


def stated_dates(event: TimelineEvent) -> list[dt.date]:
    """The days this event is said to have happened on. Empty when only a range is known."""
    if event.date is not None:
        return [event.date]
    return list(event.alternative_dates)


def merged_dates(members: list[TimelineEvent]) -> dict[str, object]:
    """Pick one date for the merged event.

    An exact day beats a range because it is more precise. Two different days are a
    conflict to report, not a choice to make.
    """
    fields: dict[str, object] = {
        "date": None,
        "date_precision": "unknown",
        "date_earliest": None,
        "date_latest": None,
        "alternative_dates": [],
    }
    stated = sorted({date for member in members for date in stated_dates(member)})
    if len(stated) == 1:
        fields |= {"date": stated[0], "date_precision": "exact"}
    elif len(stated) > 1:
        fields |= {"date_precision": "conflicting", "alternative_dates": stated}
    elif spans := [span for span in (date_range(member) for member in members) if span]:
        earliest, latest = min(spans, key=lambda span: span[1] - span[0])
        fields |= {"date_precision": "approximate", "date_earliest": earliest, "date_latest": latest}
    return fields


def merged_reasons(members: list[TimelineEvent], date_precision: str) -> list[str]:
    """Review reasons for the merged event.

    Date reasons are worked out again from the new date; the rest are kept.
    """
    reasons = []
    if date_precision in DATE_REVIEW_REASONS:
        reasons.append(DATE_REVIEW_REASONS[date_precision])
    for member in members:
        for reason in member.review_reasons:
            if reason not in reasons and reason not in DATE_REVIEW_REASONS.values():
                reasons.append(reason)
    return reasons


class EventMatcher(Protocol):
    """Answers the one question this module cannot: are these two records of the same event?"""

    async def is_same(self, pair: EventPair) -> MatchResult: ...
