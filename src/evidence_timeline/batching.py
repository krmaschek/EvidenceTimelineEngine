"""Split documents into batches so that every source line is sent exactly once.

A document stays whole if it fits into the character budget. A longer document
is cut between lines, never inside a line, because citations point at whole
lines. A single line longer than the budget gets a batch of its own.
"""

from evidence_timeline.models import Batch, DocumentSpan, SourceDocument


def text_size(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)  # +1 for the line break


def split_document(document: SourceDocument, max_chars: int) -> list[DocumentSpan]:
    spans = []
    lines: list[str] = []
    first_line = 1
    size = 0
    for line_number, line in enumerate(document.lines, start=1):
        if lines and size + len(line) + 1 > max_chars:
            spans.append(DocumentSpan(document_id=document.document_id, first_line=first_line, lines=lines))
            lines, first_line, size = [], line_number, 0
        lines.append(line)
        size += len(line) + 1
    if lines:
        spans.append(DocumentSpan(document_id=document.document_id, first_line=first_line, lines=lines))
    return spans


def build_batches(case_id: str, documents: list[SourceDocument], max_chars: int) -> list[Batch]:
    """Pack the document spans, in order, into batches of at most max_chars characters."""
    groups: list[list[DocumentSpan]] = []
    size = 0
    for document in documents:
        for span in split_document(document, max_chars):
            if not groups or size + text_size(span.lines) > max_chars:
                groups.append([])
                size = 0
            groups[-1].append(span)
            size += text_size(span.lines)

    return [
        Batch(batch_id=f"{case_id}-batch-{number:03d}", spans=spans)
        for number, spans in enumerate(groups, start=1)
    ]


def check_coverage(documents: list[SourceDocument], batches: list[Batch]) -> None:
    """Stop if a source line is missing from the batches or appears twice."""
    expected = [(doc.document_id, n) for doc in documents for n in range(1, len(doc.lines) + 1)]
    batched = [
        (span.document_id, n) for batch in batches for span in batch.spans for n, _ in span.numbered_lines()
    ]
    if sorted(batched) != sorted(expected):
        raise RuntimeError("The batches do not contain every source line exactly once")
