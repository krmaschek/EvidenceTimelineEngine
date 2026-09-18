"""Event extraction through an OpenAI-compatible chat API (OpenRouter by default).

This is the only file that knows about the provider. Another OpenAI-compatible
provider only needs different LLM_* settings.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, SecretStr

from evidence_timeline.extractors import ExtractionError, ExtractionResult
from evidence_timeline.models import Batch, ExtractionResponse, ExtractorInfo
from evidence_timeline.prompts import build_messages

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 8_000
# Worth retrying: timeout, rate limit and temporary server problems.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMConfig(BaseModel):
    base_url: str
    api_key: SecretStr  # printed as ********
    model: str
    timeout_seconds: float
    max_attempts: int
    max_concurrent_requests: int


class LLMExtractor:
    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.AsyncBaseTransport | None = None,  # tests pass a fake server here
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,  # tests pass a function that does not wait
    ) -> None:
        self.config = config
        self.sleep = sleep
        # Limits how many requests are running at the same time, across all batches.
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
            timeout=None,  # each attempt is limited with asyncio.timeout instead
            transport=transport,
        )
        self.info = ExtractorInfo(
            kind="llm",
            model=config.model,
            settings={
                "base_url": config.base_url,
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timeout_seconds": config.timeout_seconds,
                "max_attempts": config.max_attempts,
                "max_concurrent_requests": config.max_concurrent_requests,
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def extract_events(self, batch: Batch) -> ExtractionResult:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": build_messages(batch),
            # Ask the provider to force the answer into our JSON schema.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "timeline_events",
                    "strict": True,
                    "schema": ExtractionResponse.model_json_schema(),
                },
            },
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if "openrouter.ai" in self.config.base_url:
            # Only use providers that support every parameter above, so the schema is enforced.
            request["provider"] = {"require_parameters": True}

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                async with self.semaphore:
                    async with asyncio.timeout(self.config.timeout_seconds):
                        response = await self.client.post("chat/completions", json=request)
                response.raise_for_status()

                reply = response.json()
                answer = reply["choices"][0]["message"]["content"]
                events = ExtractionResponse.model_validate_json(answer).events
                return ExtractionResult(
                    events=events, attempts=attempt, model=reply.get("model"), usage=reply.get("usage")
                )

            except httpx.HTTPStatusError as exc:
                error = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise ExtractionError(error, attempt)  # e.g. a wrong API key: retrying will not help

            except (TimeoutError, httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
                # No answer in time, a connection problem, or a reply we cannot read.
                error = repr(exc)[:300]

            if attempt < self.config.max_attempts:
                await self.sleep(2 ** (attempt - 1) + random.random())  # 1 s, 2 s, 4 s, ... plus up to 1 s jitter

        raise ExtractionError(f"failed after {self.config.max_attempts} attempts: {error}", self.config.max_attempts)
