"""The matcher is tested against a fake HTTP server (httpx.MockTransport); nothing goes over the network."""

import asyncio
import json

import httpx
import pytest
from helpers import make_quote, make_timeline_event
from pydantic import SecretStr

from evidence_timeline.llm_extractor import LLMConfig
from evidence_timeline.llm_matcher import LLMMatcher
from evidence_timeline.models import EventPair

CONFIG = LLMConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr("sk-test"),
    model="vendor/model-x",
    timeout_seconds=5,
    max_attempts=1,
    max_concurrent_requests=2,
)
PAIR = EventPair(
    a=make_timeline_event("first", evidence=[make_quote(9, "An X-ray was performed on 4 March 2025.", "B01")]),
    b=make_timeline_event("second", evidence=[make_quote(9, "The prior X-ray, on 4 March 2025.", "B03")]),
)


def reply(same_event: bool = True, reason: str = "B03 refers back to the X-ray in B01") -> httpx.Response:
    answer = json.dumps({"same_event": same_event, "reason": reason})
    body = {"choices": [{"message": {"content": answer}}], "usage": {"total_tokens": 42, "cost": 0.001}}
    return httpx.Response(200, json=body)


def decide(response: httpx.Response, pair: EventPair = PAIR):
    requests: list[httpx.Request] = []

    def server(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    matcher = LLMMatcher(CONFIG, transport=httpx.MockTransport(server))
    decision = asyncio.run(matcher.is_same(pair))
    return decision, json.loads(requests[0].content)


def test_the_request_sends_both_quotes_and_asks_for_strict_json():
    _, sent = decide(reply())

    data = sent["messages"][1]["content"]
    assert "An X-ray was performed on 4 March 2025." in data
    assert "The prior X-ray, on 4 March 2025." in data
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["model"] == "vendor/model-x"


def test_a_yes_is_read_back_with_its_reason():
    decision, _ = decide(reply())

    assert decision.same_event
    assert decision.reason == "B03 refers back to the X-ray in B01"


def test_a_no_is_read_back():
    decision, _ = decide(reply(same_event=False, reason="two separate sessions"))

    assert not decision.same_event


def test_the_cost_reported_by_the_provider_is_kept():
    decision, _ = decide(reply())

    assert decision.usage == {"total_tokens": 42, "cost": 0.001}


def test_a_failed_request_raises_so_the_caller_can_retry_or_give_up():
    with pytest.raises(httpx.HTTPStatusError):
        decide(httpx.Response(500, text="server error"))
