"""Ask the LLM whether two events are the same, with one small request per pair.

There is no retry loop here. In the Temporal worker, Temporal retries the activity;
in the CLI a failed request is not retried. Either way the pair is just not merged.
"""

import asyncio
from typing import Any

import httpx

from evidence_timeline.llm_extractor import MAX_OUTPUT_TOKENS, TEMPERATURE, LLMConfig
from evidence_timeline.models import EventPair, MatchDecision, MatchResult
from evidence_timeline.prompts import build_match_messages


class LLMMatcher:
    def __init__(self, config: LLMConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        # Limits how many requests run at the same time.
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
            timeout=None,  # each request is limited with asyncio.timeout instead
            transport=transport,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def is_same(self, pair: EventPair) -> MatchResult:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": build_match_messages(pair),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "match_decision",
                    "strict": True,
                    "schema": MatchDecision.model_json_schema(),
                },
            },
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if "openrouter.ai" in self.config.base_url:
            # Only use providers that support every parameter above, so the schema is enforced.
            request["provider"] = {"require_parameters": True}

        async with self.semaphore:
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self.client.post("chat/completions", json=request)
        response.raise_for_status()
        reply = response.json()
        answer = MatchDecision.model_validate_json(reply["choices"][0]["message"]["content"])
        return MatchResult(same_event=answer.same_event, reason=answer.reason, usage=reply.get("usage"))
