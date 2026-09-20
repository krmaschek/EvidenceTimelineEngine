"""The process that runs workflows and activities.

Temporal's server only keeps the history and hands out work; our code runs here.
Starting a second worker is how the work is spread, and how a run survives one
worker being killed.
"""

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from evidence_timeline.activities import Activities
from evidence_timeline.extractors import EventExtractor
from evidence_timeline.workflows import BuildTimelineWorkflow

DEFAULT_ADDRESS = "localhost:7233"
TASK_QUEUE = "evidence-timeline"


async def connect(address: str) -> Client:
    """Our Pydantic models cross the worker's boundary, so both sides need the same converter."""
    return await Client.connect(address, data_converter=pydantic_data_converter)


async def run_worker(
    extractor: EventExtractor, database_url: str | None, address: str, max_concurrent: int
) -> None:
    client = await connect(address)
    activities = Activities(extractor, database_url)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuildTimelineWorkflow],
        activities=[activities.plan_case, activities.extract_batch, activities.save_timeline_run],
        # The limit that really decides how many LLM calls happen at once.
        max_concurrent_activities=max_concurrent,
    )
    print(f"Worker ready on {address}, task queue {TASK_QUEUE!r}.")
    print(f"  extractor: {extractor.info.kind}, at most {max_concurrent} activities at a time")
    print(f"  database: {'yes' if database_url else 'no'}")
    print("  submit a case with: evidence-timeline submit <case folder>")
    print("  stop with Ctrl+C")
    await worker.run()
