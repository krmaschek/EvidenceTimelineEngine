"""Event extraction through an OpenAI-compatible chat API (OpenRouter by default).

This is the only file that knows about the provider. Another OpenAI-compatible
provider only needs different LLM_* settings.
"""

import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from evidence_timeline.extractors import ExtractionError, ExtractionResult
from evidence_timeline.models import Batch, ExtractionResponse, ExtractorInfo
from evidence_timeline.prompts import build_messages

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 8_000
REQUIRED_PARAMETERS = ["structured_outputs", "temperature", "max_tokens"]
# Worth retrying: timeout, rate limit and temporary server problems.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMConfig(BaseModel):
    base_url: str
    api_key: SecretStr  # printed as ********
    model: str
    timeout_seconds: float
    max_attempts: int


# The parts of the provider's reply that we use.
class Message(BaseModel):
    content: str | None = None


class Choice(BaseModel):
    message: Message


class ChatCompletion(BaseModel):
    model: str | None = None
    choices: list[Choice] = Field(min_length=1)
    usage: dict[str, Any] | None = None


class OpenAICompatibleExtractor:
    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.BaseTransport | None = None,  # tests pass a fake server here
        sleep: Callable[[float], None] = time.sleep,  # tests pass a function that does not wait
    ) -> None:
        self.config = config
        self.sleep = sleep
        self.client = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
            timeout=config.timeout_seconds,
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
            },
        )

    def supports_required_parameters(self) -> bool | None:
        """Look the model up in the provider's model list. None means the provider does not say."""
        response = self.client.get("models")
        response.raise_for_status()
        for model in response.json()["data"]:
            if model["id"] == self.config.model:
                supported = model.get("supported_parameters")
                if supported is None:
                    return None
                return all(name in supported for name in REQUIRED_PARAMETERS)
        raise ValueError(f"Model {self.config.model!r} is not offered by {self.config.base_url}")

    def extract(self, batch: Batch) -> ExtractionResult:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": build_messages(batch),
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

        response, attempts = self.post_with_retries(request)
        try:
            completion = ChatCompletion.model_validate_json(response.text)
            content = completion.choices[0].message.content or ""
            events = ExtractionResponse.model_validate_json(content).events
        except ValidationError as error:
            # Also covers output cut off at max_tokens: it is not valid JSON.
            raise ExtractionError(f"invalid model response: {error.errors()[0]['msg']}", attempts) from error

        return ExtractionResult(events=events, attempts=attempts, model=completion.model, usage=completion.usage)

    def post_with_retries(self, request: dict[str, Any]) -> tuple[httpx.Response, int]:
        """Send the request. Retry only errors that are likely to be temporary."""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.client.post("chat/completions", json=request)
                if response.status_code == 200:
                    return response, attempt
                error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise ExtractionError(error, attempt)  # e.g. wrong API key or bad request
            except httpx.RequestError as exc:  # timeout or connection problem
                error = f"{type(exc).__name__}: {exc}"

            if attempt < self.config.max_attempts:
                self.sleep(2 ** (attempt - 1))  # wait 1 s, 2 s, 4 s, ...

        raise ExtractionError(f"failed after {self.config.max_attempts} attempts: {error}", self.config.max_attempts)
