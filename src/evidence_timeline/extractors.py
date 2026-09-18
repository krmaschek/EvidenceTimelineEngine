"""The extraction interface used by the pipeline, plus a fake extractor for offline runs.

The pipeline only knows `EventExtractor`. Supporting another LLM API means
writing one more class with an `info` attribute and an `extract` method.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from evidence_timeline.models import Batch, EvidenceQuote, ExtractedEvent, ExtractorInfo


class ExtractionError(Exception):
    """A batch could not be extracted. The pipeline records it as a failed batch."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class ExtractionResult(BaseModel):
    events: list[ExtractedEvent]
    attempts: int
    model: str | None = None
    usage: dict[str, Any] | None = None


class EventExtractor(Protocol):
    info: ExtractorInfo  # stored with the run

    async def extract_events(self, batch: Batch) -> ExtractionResult: ...


class FakeExtractor:
    """Stand-in for the LLM, used to test the pipeline without an API key.

    It does not understand the text and never reads reference answers. For each
    document in a batch it returns one placeholder event that quotes the first
    non-empty line. Its output must never be scored as extraction quality.
    """

    info = ExtractorInfo(kind="fake", model=None, settings={})

    async def extract_events(self, batch: Batch) -> ExtractionResult:
        events = []
        for span in batch.spans:
            for line_number, text in span.numbered_lines():
                if text.strip():
                    events.append(placeholder_event(span.document_id, line_number, text))
                    break
        return ExtractionResult(events=events, attempts=1)


def placeholder_event(document_id: str, line_number: int, text: str) -> ExtractedEvent:
    quote = EvidenceQuote(
        document_id=document_id, line_start=line_number, line_end=line_number, quote=text, date_text=None
    )
    return ExtractedEvent(
        event_type="visit",  # a fixed placeholder, not a judgement about the text
        status="completed",
        description=f"[FAKE] placeholder for {document_id}",
        date=None,
        date_precision="unknown",
        date_earliest=None,
        date_latest=None,
        alternative_dates=[],
        evidence=[quote],
        review_reasons=["fake extractor: not a real event"],
    )
