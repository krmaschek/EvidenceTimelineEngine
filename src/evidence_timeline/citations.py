"""Check where a quote claims to come from.

A citation passes if its document and lines were part of the batch the model
saw and the quote appears in those lines. This does NOT prove that the quote
supports the event, its date or its status; that still needs a person.
"""

from evidence_timeline.models import Batch, EvidenceQuote


def collapse_whitespace(text: str) -> str:
    # Models may join cited lines with a space or a newline, so whitespace is not compared.
    return " ".join(text.split())


def check_citation(evidence: EvidenceQuote, batch: Batch) -> str | None:
    """Return what is wrong with the citation, or None if it is fine."""
    location = f"{evidence.document_id} lines {evidence.line_start}-{evidence.line_end}"
    lines = {
        number: text
        for span in batch.spans
        if span.document_id == evidence.document_id
        for number, text in span.numbered_lines()
    }
    if evidence.line_start not in lines or evidence.line_end not in lines:
        return f"{location} are not in this batch"

    cited_text = collapse_whitespace(" ".join(lines[n] for n in range(evidence.line_start, evidence.line_end + 1)))
    quote = collapse_whitespace(evidence.quote)
    if not quote or quote not in cited_text:
        return f"quote not found in {location}"
    if evidence.date_text and evidence.date_text not in evidence.quote:
        return f"date_text {evidence.date_text!r} is not part of the quote"
    return None
