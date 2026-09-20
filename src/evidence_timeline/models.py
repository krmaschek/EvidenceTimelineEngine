"""Data shapes used by the pipeline.

Pydantic checks the structure of the data and a few date rules. It cannot tell
whether an event is really supported by the documents.
"""

import datetime as dt
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal["visit", "procedure", "medication_start"]
EventStatus = Literal["completed", "planned"]
# exact: one known day | approximate: only a date range | conflicting: sources disagree | unknown: no date
DatePrecision = Literal["exact", "approximate", "conflicting", "unknown"]


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


class CaseRequest(BaseModel):
    """What one run needs to know before it starts."""

    case_dir: str  # a string, not a Path, because it travels to a worker as JSON
    max_chars: int


class CasePlan(BaseModel):
    """Everything known before any model is called: the batches, and which extractor will run them."""

    case_id: str
    extractor: ExtractorInfo
    batches: list[Batch]
    document_sha256: dict[str, str]
    total_lines: int


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

    event_type: EventType
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
        # A JSON schema cannot express these rules. Breaking them would invent or hide precision.
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
            raise ValueError(f"date fields do not fit date_precision={self.date_precision!r}")
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


class BatchReport(BaseModel):
    batch_id: str
    lines: list[str]  # e.g. ["A01:1-11", "A02:1-10"]
    status: Literal["succeeded", "failed"]
    attempts: int
    error: str | None = None
    model: str | None = None  # the model the provider says it used
    usage: dict[str, Any] | None = None  # token counts reported by the provider


class BatchOutcome(BaseModel):
    """What one batch produced. A failed batch has a report and no events."""

    report: BatchReport
    events: list[TimelineEvent]


class TimelineRun(BaseModel):
    run_id: str
    created_at: dt.datetime
    case_id: str
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
