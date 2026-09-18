"""The LLM adapter is tested against a fake HTTP server (httpx.MockTransport); nothing goes over the network."""

import asyncio
import datetime as dt
import json

import httpx
import pytest
from helpers import make_batch, make_event, make_quote
from pydantic import SecretStr

from evidence_timeline.extractors import ExtractionError
from evidence_timeline.llm_extractor import LLMConfig, LLMExtractor

CONFIG = LLMConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr("sk-test"),
    model="vendor/model-x",
    timeout_seconds=5,
    max_attempts=3,
    max_concurrent_requests=2,
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


def make_extractor(server, config: LLMConfig = CONFIG) -> tuple[LLMExtractor, list[float]]:
    waits: list[float] = []

    async def record_wait(seconds: float) -> None:
        waits.append(seconds)

    extractor = LLMExtractor(config, transport=httpx.MockTransport(server), sleep=record_wait)
    return extractor, waits


def test_request_asks_the_configured_model_for_strict_json():
    server = FakeServer(reply())
    extractor, _ = make_extractor(server)

    asyncio.run(extractor.extract_events(BATCH))

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

    asyncio.run(extractor.extract_events(BATCH))

    assert str(server.requests[0].url) == "http://localhost:8000/v1/chat/completions"
    assert "provider" not in json.loads(server.requests[0].content)


def test_reply_becomes_events_with_model_and_usage():
    extractor, _ = make_extractor(FakeServer(reply()))

    result = asyncio.run(extractor.extract_events(BATCH))

    assert result.events[0].date == dt.date(2025, 1, 6)
    assert (result.attempts, result.model, result.usage) == (1, "vendor/model-x", {"total_tokens": 42})


def test_api_key_is_not_printed_or_stored_with_the_run():
    extractor, _ = make_extractor(FakeServer())

    assert "sk-test" not in repr(CONFIG)
    assert "sk-test" not in extractor.info.model_dump_json()


def test_temporary_errors_are_retried_with_growing_waits():
    server = FakeServer(httpx.Response(503), httpx.ConnectError("connection reset"), reply())
    extractor, waits = make_extractor(server)

    result = asyncio.run(extractor.extract_events(BATCH))

    assert result.attempts == 3
    assert [int(wait) for wait in waits] == [1, 2]  # the part after the decimal point is random jitter


def test_slow_answers_time_out_and_are_retried():
    requests = []

    async def slow_first_answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            await asyncio.sleep(10)  # longer than the timeout below
        return reply()

    extractor, _ = make_extractor(slow_first_answer, CONFIG.model_copy(update={"timeout_seconds": 0.05}))

    result = asyncio.run(extractor.extract_events(BATCH))

    assert result.attempts == 2


def test_retries_stop_after_max_attempts():
    server = FakeServer(httpx.ConnectError("down"), httpx.ConnectError("down"), httpx.ConnectError("down"))
    extractor, waits = make_extractor(server)

    with pytest.raises(ExtractionError, match="failed after 3 attempts: ConnectError") as error:
        asyncio.run(extractor.extract_events(BATCH))

    assert error.value.attempts == 3
    assert len(waits) == 2


@pytest.mark.parametrize("status", [400, 401, 404])
def test_permanent_errors_are_not_retried(status):
    server = FakeServer(httpx.Response(status, text="nope"))
    extractor, waits = make_extractor(server)

    with pytest.raises(ExtractionError, match=f"HTTP {status}: nope"):
        asyncio.run(extractor.extract_events(BATCH))

    assert len(server.requests) == 1
    assert waits == []


def test_invalid_output_is_retried_with_the_same_request():
    server = FakeServer(reply(content="not json"), reply())
    extractor, _ = make_extractor(server)

    result = asyncio.run(extractor.extract_events(BATCH))

    assert result.attempts == 2
    assert server.requests[0].content == server.requests[1].content


@pytest.mark.parametrize(
    "bad_reply",
    [
        lambda: reply(content="not json"),
        lambda: reply(content=None),
        lambda: reply(content=EVENTS_JSON.replace('"exact"', '"approximate"')),  # breaks the date rules
        lambda: httpx.Response(200, json={"choices": []}),
    ],
    ids=["not json", "no content", "breaks date rules", "no choices"],
)
def test_invalid_output_fails_the_batch_after_max_attempts(bad_reply):
    server = FakeServer(bad_reply(), bad_reply(), bad_reply())
    extractor, _ = make_extractor(server)

    with pytest.raises(ExtractionError, match="failed after 3 attempts"):
        asyncio.run(extractor.extract_events(BATCH))

    assert len(server.requests) == 3


def test_no_more_than_max_concurrent_requests_run_at_once():
    running = 0
    most_at_once = 0

    async def busy_server(request: httpx.Request) -> httpx.Response:
        nonlocal running, most_at_once
        running += 1
        most_at_once = max(most_at_once, running)
        await asyncio.sleep(0.01)
        running -= 1
        return reply()

    extractor, _ = make_extractor(busy_server)  # CONFIG allows 2 requests at once

    async def extract_six_batches():
        return await asyncio.gather(*[extractor.extract_events(BATCH) for _ in range(6)])

    results = asyncio.run(extract_six_batches())

    assert len(results) == 6
    assert most_at_once == 2
