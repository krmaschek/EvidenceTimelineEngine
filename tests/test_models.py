import datetime as dt
import json
from typing import Any

import pytest
from pydantic import ValidationError

from evidence_timeline.models import ExtractedEvent, ExtractionResponse


def event_json(**changes: Any) -> dict[str, Any]:
    event = {
        "event_type": "procedure",
        "status": "planned",
        "description": "Right knee MRI",
        "date": "2025-05-02",
        "date_precision": "exact",
        "date_earliest": None,
        "date_latest": None,
        "alternative_dates": [],
        "evidence": [
            {
                "document_id": "D01",
                "line_start": 8,
                "line_end": 8,
                "quote": "MRI booked for 2 May 2025.",
                "date_text": "2 May 2025",
            }
        ],
        "review_reasons": [],
    }
    event.update(changes)
    return event


def parse(event: dict[str, Any]) -> ExtractedEvent:
    return ExtractionResponse.model_validate_json(json.dumps({"events": [event]})).events[0]


def test_model_output_becomes_typed_events():
    event = parse(event_json())

    assert event.status == "planned"
    assert event.date == dt.date(2025, 5, 2)
    assert event.evidence[0].date_text == "2 May 2025"


def test_uncertain_dates_are_accepted_without_a_precise_date():
    approximate = parse(
        event_json(date=None, date_precision="approximate", date_earliest="2025-03-01", date_latest="2025-03-31")
    )
    conflicting = parse(
        event_json(date=None, date_precision="conflicting", alternative_dates=["2025-04-09", "2025-04-10"])
    )

    assert approximate.date is None
    assert approximate.date_latest == dt.date(2025, 3, 31)
    assert conflicting.alternative_dates == [dt.date(2025, 4, 9), dt.date(2025, 4, 10)]


@pytest.mark.parametrize(
    "changes",
    [
        {"date_precision": "approximate"},  # a precise date marked as approximate
        {"date": None},  # "exact" without a date
        {"date": None, "date_precision": "approximate", "date_earliest": "2025-03-31", "date_latest": "2025-03-01"},
        {"date": None, "date_precision": "conflicting", "alternative_dates": ["2025-04-09"]},
        {"date_precision": "conflicting", "alternative_dates": ["2025-04-09", "2025-04-10"]},  # conflict, yet a date
    ],
)
def test_dates_that_do_not_fit_the_precision_are_cleared_and_flagged(changes):
    event = parse(event_json(**changes))

    assert event.date_precision == "unknown"
    assert (event.date, event.date_earliest, event.date_latest, event.alternative_dates) == (None, None, None, [])
    assert event.review_reasons[-1].startswith("dates removed: they did not fit date_precision=")
    assert event.description == "Right knee MRI"  # the rest of the event is kept


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": 0.9},
        # An unknown event_type is not rejected here: each dataset has its own types, and the
        # schema sent to the model limits them (see test_llm_extractor).
        {"status": "done"},
        {"date": "6 January 2025"},
        {"evidence": []},
    ],
)
def test_malformed_events_are_rejected(changes):
    with pytest.raises(ValidationError):
        parse(event_json(**changes))


def test_schema_sent_to_the_model_is_strict():
    schema = ExtractionResponse.model_json_schema()

    for definition in [schema, *schema["$defs"].values()]:
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])
    assert "allOf" not in json.dumps(schema)
