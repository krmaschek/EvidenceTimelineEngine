"""Check that the files of this dataset fit together: documents, checksums and answer keys.

It runs no model. It needs only Python's standard library and works from any folder:

    python validate_dataset.py                  print the report
    python validate_dataset.py --write-report   also save it to VALIDATION_REPORT.json

The first check that fails stops the script with an error that names it.
"""

import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST_RECORD_LINE = 8  # lines 1-7 are the header: title, banner, case, document ID and creation date
EVENT_TYPES = {"visit", "procedure", "medication_start"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_documents(manifest: dict) -> dict[str, list[str]]:
    """Check every document against the manifest. Returns the lines of each document, by its ID."""
    documents = manifest["documents"]
    ids = [document["document_id"] for document in documents]
    require(len(ids) == len(set(ids)) == 14, "Document IDs/count invalid")

    texts = {}
    for document in documents:
        path = ROOT / document["path"]
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == document["sha256"], f"Checksum: {path}")
        text = data.decode("utf-8")
        require(f"Case: {document['case_id']}\n" in text, "Case metadata mismatch")
        require(f"Document ID: {document['document_id']}\n" in text, "Document metadata mismatch")
        date.fromisoformat(document["record_created"])
        texts[document["document_id"]] = text.splitlines()

    listed = {ROOT / document["path"] for document in documents}
    require(set(ROOT.glob("inputs/*/*.md")) == listed, "Unlisted input files")
    return texts


def check_source(source: dict, case: str, texts: dict[str, list[str]], coverage: Counter) -> None:
    """Check that a quote is exactly on the lines it cites, and count those lines as covered."""
    document_id = source["document_id"]
    require(document_id in texts, "Unknown source document")
    require(document_id[0] == case, "Cross-case citation")  # e.g. A01 belongs to case A

    first = source["line_start"]
    last = source["line_end"]
    lines = texts[document_id]
    require(FIRST_RECORD_LINE <= first <= last <= len(lines), "Invalid narrative line range")
    require("\n".join(lines[first - 1 : last]) == source["quote"], "Quote/location mismatch")
    if source["date_text"] is not None:
        require(source["date_text"] in source["quote"], "Original date text missing from quote")

    for line_number in range(first, last + 1):
        coverage[document_id, line_number] += 1


def check_dates(event: dict) -> None:
    """Check that the date fields fit the date's precision, and that an uncertain date is flagged."""
    precision = event["date_precision"]
    if precision == "day":
        date.fromisoformat(event["date"])
        require(event["date_interval"] is None and not event["alternative_dates"], "Invalid exact date")
    elif precision == "month":
        require(event["date"] is None and not event["alternative_dates"], "False precision")
        start = date.fromisoformat(event["date_interval"]["start"])
        end = date.fromisoformat(event["date_interval"]["end"])
        same_month = start.strftime("%Y-%m") == end.strftime("%Y-%m")
        require(start <= end and same_month, "Invalid interval")
        require("approximate_date" in event["review_reasons"], "Unflagged uncertainty")
    elif precision == "conflicting":
        require(event["date"] is None and event["date_interval"] is None, "Silently resolved conflict")
        require(len(set(event["alternative_dates"])) >= 2, "Missing conflict alternatives")
        for value in event["alternative_dates"]:
            date.fromisoformat(value)
        require("conflicting_dates" in event["review_reasons"], "Unflagged conflict")
    else:
        raise ValueError("Unknown date precision")


def check_case(case: str, texts: dict[str, list[str]], coverage: Counter) -> tuple[list[dict], int]:
    """Check one case's answer key. Returns its events and the number of excluded lines."""
    gold = read_json(ROOT / "gold" / f"case_{case}.json")
    require(gold["case_id"] == case and len(gold["events"]) == 8, "Case/count mismatch")

    for event in gold["events"]:
        require(event["case_id"] == case and event["event_id"].startswith(f"{case}-E"), "Event case mismatch")
        require(event["event_type"] in EVENT_TYPES, "Bad event type")
        require(event["status"] in {"planned", "completed"}, "Bad status")
        require(event["sources"], "Missing evidence")
        require(event["needs_review"] == bool(event["review_reasons"]), "Review flag mismatch")
        require("confidence" not in event, "Unexpected confidence score")
        for source in event["sources"]:
            check_source(source, case, texts, coverage)
        check_dates(event)

    # A record line without an event is excluded on purpose, with a reason.
    for exclusion in gold["exclusions"]:
        require(exclusion["case_id"] == case and exclusion["reason"], "Invalid exclusion")
        check_source(exclusion["source"], case, texts, coverage)

    # C05 is an administrative note, so no event may cite it.
    expected_zero = ["C05"] if case == "C" else []
    require(gold["zero_event_documents"] == expected_zero, "Zero-event annotation mismatch")
    for event in gold["events"]:
        for source in event["sources"]:
            require(source["document_id"] not in expected_zero, "Zero-event doc has event")

    return gold["events"], len(gold["exclusions"])


def validate() -> dict:
    manifest = read_json(ROOT / "manifest.json")
    require(manifest["splits"] == {"development": ["A", "B"], "holdout": ["C"]}, "Invalid split")
    texts = check_documents(manifest)

    coverage: Counter = Counter()  # how often each (document ID, line number) is cited
    events = []
    exclusions = 0
    for case in ["A", "B", "C"]:
        case_events, case_exclusions = check_case(case, texts, coverage)
        events += case_events
        exclusions += case_exclusions

    event_ids = [event["event_id"] for event in events]
    require(len(event_ids) == len(set(event_ids)) == 24, "Event IDs/count invalid")

    # Every non-empty record line belongs to exactly one event or exclusion.
    for document_id, lines in texts.items():
        for line_number, line in enumerate(lines, start=1):
            if line_number >= FIRST_RECORD_LINE and line.strip():
                message = f"Unaccounted or multiply labelled narrative: {document_id}:{line_number}"
                require(coverage[document_id, line_number] == 1, message)

    # The difficult events the dataset was made for must not change by accident.
    by_id = {event["event_id"]: event for event in events}
    planned = {event["event_id"] for event in events if event["status"] == "planned"}
    require(planned == {"C-E03", "C-E08"}, "Planned status changed")
    march = {"start": "2025-03-01", "end": "2025-03-31"}
    require(by_id["C-E05"]["date_interval"] == march, "Month bounds changed")
    require(by_id["C-E06"]["alternative_dates"] == ["2025-04-14", "2025-04-15"], "Conflict dates changed")
    repeated = [event for event in events if len(event["sources"]) > 1]
    repeated_ids = {event["event_id"] for event in repeated}
    expected_repeated = {"B-E02", "B-E03", "B-E04", "B-E07", "C-E03", "C-E06"}
    require(repeated_ids == expected_repeated, "Repeated mention groups changed")

    return {
        "status": "passed",
        "dataset_version": manifest["dataset_version"],
        "documents": len(manifest["documents"]),
        "events": len(events),
        "completed_events": sum(event["status"] == "completed" for event in events),
        "planned_events": len(planned),
        "evidence_mentions": sum(len(event["sources"]) for event in events),
        "events_with_multiple_sources": len(repeated),
        "explicit_exclusions": exclusions,
        "accounted_narrative_lines": len(coverage),
        "conflicting_date_events": sum(event["date_precision"] == "conflicting" for event in events),
        "approximate_date_events": sum(event["date_precision"] == "month" for event in events),
        "zero_event_documents": ["C05"],
        "checks": [
            "input checksums and inventory",
            "unique IDs and case isolation",
            "exact quotes and line locations",
            "complete narrative annotation coverage",
            "date shapes and uncertainty flags",
            "event counts, status and repeated-mention regression checks",
        ],
        "limitations": (
            "Structural and annotation-consistency checks; "
            "not independent clinical validation or LLM extraction evaluation."
        ),
    }


def main() -> None:
    report = json.dumps(validate(), indent=2) + "\n"
    if "--write-report" in sys.argv:
        (ROOT / "VALIDATION_REPORT.json").write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")


if __name__ == "__main__":
    main()
