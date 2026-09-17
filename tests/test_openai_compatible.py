"""The LLM adapter is tested against a fake HTTP server (httpx.MockTransport); nothing goes over the network."""

import datetime as dt
import json

import httpx
import pytest
from helpers import make_batch, make_event, make_quote
from pydantic import SecretStr

from evidence_timeline.extractors import ExtractionError
from evidence_timeline.openai_compatible import LLMConfig, OpenAICompatibleExtractor

CONFIG = LLMConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr("sk-test"),
    model="vendor/model-x",
    timeout_seconds=5,
    max_attempts=3,
)
LINE = "Patient attended a visit on 6 January 2025."
BATCH = make_batch([LINE])
EVENTS_JSON = json.dumps({"events": [make_event(evidence=[make_quote(1, LINE)]).model_dump(mode="json")]})


def reply(content: str | None = EVENTS_JSON) -> httpx.Response:
    body = {"model": "vendor/model-x", "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}}
    return httpx.Response(200, json=body)


class FakeServer:
    """Answers requests with the given replies, in order, and remembers the requests."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        next_reply = self.replies.pop(0)
        if isinstance(next_reply, Exception):
            raise next_reply
        return next_reply


def make_extractor(server: FakeServer, config: LLMConfig = CONFIG) -> tuple[OpenAICompatibleExtractor, list[float]]:
    waits: list[float] = []
    extractor = OpenAICompatibleExtractor(config, transport=httpx.MockTransport(server), sleep=waits.append)
    return extractor, waits


def test_request_asks_the_configured_model_for_strict_json():
    server = FakeServer(reply())
    extractor, _ = make_extractor(server)

    extractor.extract(BATCH)

    request = server.requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "vendor/model-x"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    assert LINE in body["messages"][1]["content"]


def test_other_providers_get_no_openrouter_settings():
    server = FakeServer(reply())
    extractor, _ = make_extractor(server, CONFIG.model_copy(update={"base_url": "http://localhost:8000/v1"}))

    extractor.extract(BATCH)

    assert str(server.requests[0].url) == "http://localhost:8000/v1/chat/completions"
    assert "provider" not in json.loads(server.requests[0].content)


def test_reply_becomes_events_with_model_and_usage():
    extractor, _ = make_extractor(FakeServer(reply()))

    result = extractor.extract(BATCH)

    assert result.events[0].date == dt.date(2025, 1, 6)
    assert (result.attempts, result.model, result.usage) == (1, "vendor/model-x", {"total_tokens": 42})


def test_api_key_is_not_printed_or_stored_with_the_run():
    extractor, _ = make_extractor(FakeServer())

    assert "sk-test" not in repr(CONFIG)
    assert "sk-test" not in extractor.info.model_dump_json()


def test_temporary_errors_are_retried():
    server = FakeServer(httpx.Response(503), httpx.ConnectError("connection reset"), reply())
    extractor, waits = make_extractor(server)

    result = extractor.extract(BATCH)

    assert result.attempts == 3
    assert waits == [1, 2]


def test_retries_stop_after_max_attempts():
    server = FakeServer(httpx.ReadTimeout("timed out"), httpx.ReadTimeout("timed out"), httpx.ReadTimeout("timed out"))
    extractor, waits = make_extractor(server)

    with pytest.raises(ExtractionError, match="failed after 3 attempts: ReadTimeout: timed out") as error:
        extractor.extract(BATCH)

    assert error.value.attempts == 3
    assert waits == [1, 2]


@pytest.mark.parametrize("status", [400, 401, 404])
def test_permanent_errors_are_not_retried(status):
    server = FakeServer(httpx.Response(status, text="nope"))
    extractor, waits = make_extractor(server)

    with pytest.raises(ExtractionError, match=f"HTTP {status}: nope"):
        extractor.extract(BATCH)

    assert len(server.requests) == 1
    assert waits == []


@pytest.mark.parametrize(
    "bad_reply",
    [
        reply(content="not json"),
        reply(content=None),
        reply(content=EVENTS_JSON.replace('"exact"', '"approximate"')),  # breaks the date rules
        httpx.Response(200, json={"choices": []}),
    ],
)
def test_invalid_replies_fail_the_batch(bad_reply):
    server = FakeServer(bad_reply)
    extractor, _ = make_extractor(server)

    with pytest.raises(ExtractionError, match="invalid model response"):
        extractor.extract(BATCH)

    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("listed_model", "expected"),
    [
        ({"id": "vendor/model-x", "supported_parameters": ["structured_outputs", "temperature", "max_tokens"]}, True),
        ({"id": "vendor/model-x", "supported_parameters": ["temperature", "max_tokens"]}, False),
        ({"id": "vendor/model-x"}, None),
    ],
)
def test_capability_check_reads_the_model_list(listed_model, expected):
    server = FakeServer(httpx.Response(200, json={"data": [{"id": "other/model"}, listed_model]}))
    extractor, _ = make_extractor(server)

    assert extractor.supports_required_parameters() is expected
    assert str(server.requests[0].url) == "https://openrouter.ai/api/v1/models"


def test_capability_check_rejects_an_unknown_model():
    extractor, _ = make_extractor(FakeServer(httpx.Response(200, json={"data": [{"id": "other/model"}]})))

    with pytest.raises(ValueError, match="not offered"):
        extractor.supports_required_parameters()
