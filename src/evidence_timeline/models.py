"""Data shapes used by the pipeline.

Pydantic checks the structure of the data and a few date rules. It cannot tell
whether an event is really supported by the documents.
"""

import datetime as dt
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventStatus = Literal["completed", "planned"]
# exact: one known day | approximate: only a date range | conflicting: sources disagree | unknown: no date
DatePrecision = Literal["exact", "approximate", "conflicting", "unknown"]
# The steps of one workflow run, in order.
Stage = Literal["planning", "extracting", "merging", "saving", "done"]


# --- Input -------------------------------------------------------------------


class SourceDocument(BaseModel):
    document_id: str
    sha256: str
    lines: list[str]  # lines[0] is line 1


class DocumentSpan(BaseModel):
    """Consecutive lines of one document, with their original line numbers."""

    document_id: str
    first_line: int
    lines: list[str]

    @property
    def last_line(self) -> int:
        return self.first_line + len(self.lines) - 1

    def numbered_lines(self) -> list[tuple[int, str]]:
        return list(enumerate(self.lines, start=self.first_line))


class Batch(BaseModel):
    batch_id: str
    spans: list[DocumentSpan]

    @property
    def line_count(self) -> int:
        return sum(len(span.lines) for span in self.spans)

    def line_labels(self) -> list[str]:
        """The lines this batch covers, e.g. ["A01:1-11", "A02:1-10"]."""
        return [f"{span.document_id}:{span.first_line}-{span.last_line}" for span in self.spans]


class ExtractorInfo(BaseModel):
    kind: Literal["fake", "llm"]
    model: str | None
    settings: dict[str, Any]


class EventTypeDefinition(BaseModel):
    name: str  # e.g. "visit"
    description: str  # finishes the sentence '"visit" for ...', e.g. "visits and assessments"


class Domain(BaseModel):
    """What counts as an event in one dataset. datasets.py reads it from the dataset folder."""

    name: str  # e.g. "clinical"
    scope: str  # the extraction scope: which events to extract and which to leave out
    event_types: list[EventTypeDefinition]
    identifiers: str  # what descriptions should keep, e.g. "body site, clinic or reference codes"


class CaseRequest(BaseModel):
    """What one run needs to know before it starts."""

    case_dir: str  # a string, not a Path, because it travels to a worker as JSON
    max_chars: int


class CasePlan(BaseModel):
    """Everything known before any model is called: the batches, the domain, and which extractor will run them."""

    case_id: str
    domain: Domain
    extractor: ExtractorInfo
    batches: list[Batch]
    document_sha256: dict[str, str]
    total_lines: int


class BatchTask(BaseModel):
    """One batch and the domain to extract it with. An activity takes one argument, so they travel together."""

    batch: Batch
    domain: Domain


# --- What the LLM must return ------------------------------------------------------
# The JSON schema sent to the model is generated from these classes, so every
# field is required and unknown fields are rejected. There is no confidence
# score: uncertainty is expressed with date_precision and review_reasons.


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    line_start: int
    line_end: int
    quote: str  # copied exactly from the cited lines
    date_text: str | None  # the original date wording inside the quote


class ExtractedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str  # one of the domain's event types; the schema sent to the model lists them
    status: EventStatus
    description: str
    date: dt.date | None
    date_precision: DatePrecision
    date_earliest: dt.date | None
    date_latest: dt.date | None
    alternative_dates: list[dt.date]
    evidence: list[EvidenceQuote] = Field(min_length=1)
    review_reasons: list[str]

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        # A JSON schema cannot express these rules. When the model breaks them, its dates cannot be
        # trusted: they are cleared and the event is flagged, instead of losing the whole answer.
        no_range = self.date_earliest is None and self.date_latest is None
        if self.date_precision == "exact":
            ok = self.date is not None and no_range and not self.alternative_dates
        elif self.date_precision == "approximate":
            ok = (
                self.date is None
                and not self.alternative_dates
                and self.date_earliest is not None
                and self.date_latest is not None
                and self.date_earliest <= self.date_latest
            )
        elif self.date_precision == "conflicting":
            ok = self.date is None and no_range and len(set(self.alternative_dates)) >= 2
        else:
            ok = self.date is None and no_range and not self.alternative_dates
        if not ok:
            self.review_reasons.append(f"dates removed: they did not fit date_precision={self.date_precision!r}")
            self.date_precision = "unknown"
            self.date = self.date_earliest = self.date_latest = None
            self.alternative_dates = []
        return self


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[ExtractedEvent]


# --- Output ------------------------------------------------------------------------


class TimelineEvent(ExtractedEvent):
    """An extracted event after the pipeline checked it.

    review_reasons holds the pipeline's reasons followed by the model's own.
    """

    event_id: str  # "<batch_id>-eNN", unique within one run
    batch_id: str
    citation_errors: list[str]  # empty when every quote was found at its cited lines
    needs_review: bool
    merged_from: list[str] = []  # the event_ids merged into this one; empty when nothing was merged


class EventPair(BaseModel):
    """Two events that might describe the same thing. An activity takes one argument, so they travel together."""

    a: TimelineEvent
    b: TimelineEvent


class MatchDecision(BaseModel):
    """The answer to one pair. The reason is kept so a merge can be explained afterwards."""

    model_config = ConfigDict(extra="forbid")

    same_event: bool
    reason: str


class MatchResult(BaseModel):
    """The matcher's answer and what the request cost. MatchDecision is the schema the model fills in,
    so the usage cannot go there."""

    same_event: bool
    reason: str
    usage: dict[str, Any] | None = None


class PairDecision(BaseModel):
    """The matcher's answer to one pair, kept in the run so every merge can be explained."""

    a: str  # event_id
    b: str  # event_id
    same_event: bool
    reason: str
    usage: dict[str, Any] | None = None  # token counts and cost reported by the provider; None in older runs


class BatchReport(BaseModel):
    batch_id: str
    lines: list[str]  # e.g. ["A01:1-11", "A02:1-10"]
    status: Literal["succeeded", "failed"]
    attempts: int
    error: str | None = None
    model: str | None = None  # the model the provider says it used
    provider: str | None = None  # the host that answered, as OpenRouter reports it
    usage: dict[str, Any] | None = None  # token counts reported by the provider


class BatchOutcome(BaseModel):
    """What one batch produced. A failed batch has a report and no events."""

    report: BatchReport
    events: list[TimelineEvent]


class TimelineRun(BaseModel):
    run_id: str
    created_at: dt.datetime
    case_id: str
    domain: str | None = None  # the domain's name; None in runs made before domains existed
    status: Literal["completed", "partial", "failed"]
    notes: list[str]
    extractor: ExtractorInfo
    max_chars_per_batch: int
    prompt_sha256: str
    document_sha256: dict[str, str]
    total_lines: int
    lines_in_successful_batches: int
    batches: list[BatchReport]
    events: list[TimelineEvent]
    pair_decisions: list[PairDecision] = []  # empty in fake runs and in runs made before they were kept


class Progress(BaseModel):
    """How far a workflow run has got. The workflow's progress query returns it."""

    stage: Stage
    total_batches: int  # 0 until the case is planned
    finished_batches: int  # succeeded or failed; either way no longer running
