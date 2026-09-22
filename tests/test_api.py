"""The HTTP API, called in-process against Temporal's test server.

httpx's ASGITransport hands each request straight to the FastAPI app, so no web
server is started. The app's lifespan does not run that way, so the test gives
the app its Temporal client itself.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from helpers import CASE_A_DIR, temporal_worker
from temporalio.client import Client

from evidence_timeline import api
from evidence_timeline.extractors import ExtractionResult, FakeExtractor
from evidence_timeline.models import Batch, Domain


class WaitingExtractor:
    """Behaves like the fake extractor, but only once the test lets it go on."""

    info = FakeExtractor.info

    def __init__(self) -> None:
        self.go = asyncio.Event()

    async def extract_events(self, batch: Batch, domain: Domain) -> ExtractionResult:
        await self.go.wait()
        return await FakeExtractor().extract_events(batch, domain)


@asynccontextmanager
async def api_client(temporal: Client) -> AsyncIterator[httpx.AsyncClient]:
    api.app.state.temporal = temporal
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def wait_until(
    http: httpx.AsyncClient, job_id: str, condition: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    """Poll the job, as a client would, until the condition holds."""
    for _ in range(100):
        job = (await http.get(f"/jobs/{job_id}")).json()
        if condition(job):
            return job
        await asyncio.sleep(0.1)
    raise AssertionError(f"the job never got there: {job}")


def test_a_job_can_be_followed_from_start_to_timeline():
    asyncio.run(follow_a_job())


async def follow_a_job() -> None:
    extractor = WaitingExtractor()
    async with temporal_worker(extractor) as temporal, api_client(temporal) as http:
        response = await http.post("/jobs", json={"case_dir": str(CASE_A_DIR), "max_chars": 500})  # 4 batches
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        # Every batch has started, and each one waits for the extractor.
        job = await wait_until(http, job_id, lambda job: job["progress"]["stage"] == "extracting")
        assert job["status"] == "RUNNING"
        assert job["progress"] == {"stage": "extracting", "total_batches": 4, "finished_batches": 0}
        assert (await http.get(f"/jobs/{job_id}/timeline")).status_code == 409

        extractor.go.set()
        job = await wait_until(http, job_id, lambda job: job["status"] == "COMPLETED")
        assert job["progress"] == {"stage": "done", "total_batches": 4, "finished_batches": 4}

        response = await http.get(f"/jobs/{job_id}/timeline")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert len(response.json()["events"]) == 4


def test_an_unknown_job_is_not_found():
    asyncio.run(ask_about_an_unknown_job())


async def ask_about_an_unknown_job() -> None:
    async with temporal_worker(FakeExtractor()) as temporal, api_client(temporal) as http:
        assert (await http.get("/jobs/no-such-job")).status_code == 404
        assert (await http.get("/jobs/no-such-job/timeline")).status_code == 404
