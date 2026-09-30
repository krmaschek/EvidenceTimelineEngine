"""Turn the EventsMatter corpus into a dataset folder the pipeline can read.

EventsMatter is 30 decisions of the European Court of Human Rights, with events (what, who,
when) annotated by legal experts in GATE's XML format. README.md in this folder has the source,
the license and the choices made here. The script imports nothing from the pipeline.

    uv run python converters/eventsmatter/convert.py --output datasets/eventsmatter_v1

It writes:

    inputs/case_<id>/<id>.md   one decision, one paragraph per line
    gold/case_<id>.json        the annotated events, for the evaluator
    manifest.json              the source, the splits and the counts

domain.json and scope.md, the dataset's definition of an event, are written by hand and left alone.
"""

import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CORPUS_URL = "https://zenodo.org/records/4032617/files/EventMattersCorpus.zip?download=1"
CORPUS_MD5 = "a47d63917da26ec167615c460821cdd0"  # published on the Zenodo page

# The corpus is split into train, dev and test. Nothing is trained here, so train is used for
# development, and dev and test together are the holdout.
HOLDOUT_SPLITS = {"dev", "test"}


@dataclass
class Annotation:
    type: str  # e.g. "p" for a paragraph, or "Event", "Event_what", "Event_who", "Event_when"
    start: int  # character offsets into the document's text
    end: int
    features: dict[str, str]  # e.g. {"id": "14", "type": "procedure"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="the dataset folder to write")
    args = parser.parse_args()
    output: Path = args.output

    corpus = zipfile.ZipFile(io.BytesIO(download_corpus()))
    (output / "gold").mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[str]] = {"development": [], "holdout": []}
    cases = []
    for name in sorted(corpus.namelist()):
        # e.g. "EventMattersCorpus/annotated/train/CASE OF HOINESS v. NORWAY.xml"
        if not name.startswith("EventMattersCorpus/annotated/") or not name.endswith(".xml"):
            continue
        source_split = name.split("/")[2]  # "train", "dev" or "test"
        case_id = make_case_id(name)

        lines, events = convert_decision(corpus.read(name), case_id)
        write_case(output, case_id, lines, events)

        split = "holdout" if source_split in HOLDOUT_SPLITS else "development"
        splits[split].append(case_id)
        cases.append(
            {
                "case_id": case_id,
                "source_file": name,
                "source_split": source_split,
                "lines": len(lines),
                "events": len(events),
            }
        )

    manifest = {
        "dataset": output.name,
        "source": {"url": CORPUS_URL, "md5": CORPUS_MD5, "license": "GPL-2.0"},
        "splits": splits,
        "cases": cases,
    }
    write_json(output / "manifest.json", manifest)
    total_events = sum(case["events"] for case in cases)
    print(f"Wrote {len(cases)} cases with {total_events} gold events to {output}")


def download_corpus() -> bytes:
    with urllib.request.urlopen(CORPUS_URL) as response:
        data = response.read()
    if hashlib.md5(data).hexdigest() != CORPUS_MD5:
        raise ValueError("The downloaded corpus is not the published version (MD5 differs)")
    return data


def make_case_id(zip_name: str) -> str:
    """For example ".../CASE OF ALTAY v. TURKEY (No. 2).xml" becomes "ALTAY_v_TURKEY_No_2"."""
    title = Path(zip_name).stem.removeprefix("CASE OF ")
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def convert_decision(xml: bytes, case_id: str) -> tuple[list[str], list[dict]]:
    """One decision's lines (one per paragraph) and its gold events."""
    text, annotation_sets = read_gate_xml(xml)
    paragraph_spans = find_paragraphs(text, annotation_sets["Original markups"])
    lines = [clean(text[start:end]) for start, end in paragraph_spans]
    # Only the annotators' consensus is used, not their separate annotations.
    events = gold_events(case_id, text, annotation_sets["consensus"], paragraph_spans)
    return lines, events


def read_gate_xml(xml: bytes) -> tuple[str, dict[str, list[Annotation]]]:
    """The document's text, and its annotations grouped by annotation set.

    GATE stores the text in pieces with a <Node id="..."/> between them. A node's id is the
    character offset where it sits, and annotations point at those offsets.
    """
    root = ET.fromstring(xml)
    text_with_nodes = root.find("TextWithNodes")
    text = text_with_nodes.text or ""
    for node in text_with_nodes:
        text += node.tail or ""  # the text that follows this node

    annotation_sets = {}
    for annotation_set in root.findall("AnnotationSet"):
        annotations = []
        for element in annotation_set.findall("Annotation"):
            features = {}
            for feature in element.findall("Feature"):
                features[feature.findtext("Name")] = feature.findtext("Value") or ""
            start = int(element.get("StartNode"))
            end = int(element.get("EndNode"))
            annotations.append(Annotation(element.get("Type"), start, end, features))
        annotation_sets[annotation_set.get("Name")] = annotations
    return text, annotation_sets


def find_paragraphs(text: str, markups: list[Annotation]) -> list[tuple[int, int]]:
    """Start and end offsets of the non-empty paragraphs, in reading order."""
    spans = []
    for annotation in markups:
        if annotation.type == "p" and text[annotation.start : annotation.end].strip():
            spans.append((annotation.start, annotation.end))
    return sorted(spans)


def clean(text: str) -> str:
    """Collapse line breaks, tabs and non-breaking spaces into single spaces."""
    return " ".join(text.split())


def gold_events(
    case_id: str, text: str, consensus: list[Annotation], paragraph_spans: list[tuple[int, int]]
) -> list[dict]:
    # An event's what, who and when are separate annotations that carry the event's id.
    parts_by_event_id = defaultdict(list)
    for annotation in consensus:
        if annotation.type in ("Event_what", "Event_who", "Event_when"):
            parts_by_event_id[annotation.features.get("id")].append(annotation)

    event_annotations = [annotation for annotation in consensus if annotation.type == "Event"]
    event_annotations.sort(key=lambda annotation: annotation.start)

    events = []
    for number, event in enumerate(event_annotations, start=1):
        parts = parts_by_event_id[event.features["id"]]
        who = part_texts(parts, "Event_who", text)
        what = part_texts(parts, "Event_what", text)
        when = part_texts(parts, "Event_when", text)

        source = {
            "document_id": case_id,  # each case has one document, named like the case
            "line_start": line_of(event.start, paragraph_spans),
            "line_end": line_of(event.end - 1, paragraph_spans),
            "quote": clean(text[event.start : event.end]),
            "date_text": "; ".join(when) or None,
        }
        description = " ".join(who + what)  # e.g. "applicant lodged two complaints"
        if not what:  # four events have no what annotated
            description = source["quote"]

        gold_event = {
            "event_id": f"{case_id}-E{number:02d}",
            "event_type": event.features["type"],  # "procedure" or "circumstance"
            "status": "completed",  # the corpus annotates only events that happened
            "description": description,
        }
        gold_event.update(date_fields(when))
        gold_event["sources"] = [source]
        events.append(gold_event)
    return events


def part_texts(parts: list[Annotation], part_type: str, text: str) -> list[str]:
    """The text of every part of one type, such as all of an event's "when" parts, in reading order."""
    texts = []
    for part in sorted(parts, key=lambda part: part.start):
        if part.type == part_type:
            texts.append(clean(text[part.start : part.end]))
    return texts


def line_of(offset: int, paragraph_spans: list[tuple[int, int]]) -> int:
    """The line, counted from 1, of the paragraph that contains a character offset."""
    for line_number, (start, end) in enumerate(paragraph_spans, start=1):
        if start <= offset < end:
            return line_number
    raise ValueError(f"character {offset} is not inside any paragraph")


def date_fields(when_texts: list[str]) -> dict:
    """The gold date of an event, in the fields the evaluator reads.

    Only a single date written in one of three plain forms is scored:
    "5 November 2010" -> that day, "November 2010" -> that month, "2010" -> that year.
    Anything else, such as "the following day", "late summer 2010" or two dates, is left
    unscored: turning it into a date would need a person to read the surrounding text.
    """
    unscored = {"date": None, "date_interval": None, "alternative_dates": [], "date_scored": False}
    if len(when_texts) != 1:
        return unscored
    wording = re.sub(r"^(on|in) ", "", when_texts[0], flags=re.IGNORECASE)  # "on 5 May 2010" -> "5 May 2010"

    day = parse_date(wording, "%d %B %Y")
    if day:
        return {"date": day.isoformat(), "date_interval": None, "alternative_dates": [], "date_scored": True}

    month = parse_date(wording, "%B %Y")
    if month:
        last_day = calendar.monthrange(month.year, month.month)[1]
        interval = {"start": month.isoformat(), "end": month.replace(day=last_day).isoformat()}
        return {"date": None, "date_interval": interval, "alternative_dates": [], "date_scored": True}

    year = parse_date(wording, "%Y")
    if year:
        interval = {"start": year.isoformat(), "end": year.replace(month=12, day=31).isoformat()}
        return {"date": None, "date_interval": interval, "alternative_dates": [], "date_scored": True}

    return unscored


def parse_date(wording: str, pattern: str) -> dt.date | None:
    """The date if the whole wording matches the pattern, otherwise None."""
    try:
        return dt.datetime.strptime(wording, pattern).date()
    except ValueError:
        return None


def write_case(output: Path, case_id: str, lines: list[str], events: list[dict]) -> None:
    case_dir = output / "inputs" / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{case_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    write_json(output / "gold" / f"case_{case_id}.json", {"case_id": case_id, "events": events})


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
