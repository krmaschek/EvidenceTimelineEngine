"""HTTP API: start a case as a job, follow its progress, fetch its timeline.

The API never reads documents or calls the LLM. It only talks to Temporal: it
starts a workflow, asks it how far it has got, and reads its result. The work
happens in the worker, so restarting the API does not touch a running job.

A job id is a workflow id. There is no jobs table, because Temporal already
keeps each job's status, progress and result.

Run with: uv run uvicorn evidence_timeline.api:app
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from temporalio.client import WorkflowExecutionStatus, WorkflowHandle
from temporalio.service import RPCError, RPCStatusCode

from evidence_timeline.models import CaseRequest, Progress, TimelineRun
from evidence_timeline.worker import DEFAULT_ADDRESS, TASK_QUEUE, connect
from evidence_timeline.workflows import BuildTimelineWorkflow


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # Temporal's own names: RUNNING, COMPLETED, FAILED, ...
    progress: Progress


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open one Temporal client when the server starts. Every request uses it."""
    app.state.temporal = await connect(os.environ.get("TEMPORAL_ADDRESS", DEFAULT_ADDRESS))
    yield


app = FastAPI(title="Evidence Timeline Engine", lifespan=lifespan)


@app.post("/jobs", status_code=202)  # 202 Accepted: the job has started, not finished
async def create_job(case: CaseRequest, request: Request) -> JobCreated:
    job_id = f"timeline-{Path(case.case_dir).name}-{uuid.uuid4().hex[:8]}"
    await request.app.state.temporal.start_workflow(BuildTimelineWorkflow.run, case, id=job_id, task_queue=TASK_QUEUE)
    return JobCreated(job_id=job_id)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JobStatus:
    handle = request.app.state.temporal.get_workflow_handle_for(BuildTimelineWorkflow.run, job_id)
    status = await get_status(handle)
    # A worker answers the query from the workflow's counters, so this needs a worker running.
    progress = await handle.query(BuildTimelineWorkflow.progress)
    return JobStatus(job_id=job_id, status=status.name, progress=progress)


@app.get("/jobs/{job_id}/timeline")
async def get_timeline(job_id: str, request: Request) -> TimelineRun:
    handle = request.app.state.temporal.get_workflow_handle_for(BuildTimelineWorkflow.run, job_id)
    status = await get_status(handle)
    if status != WorkflowExecutionStatus.COMPLETED:
        # 409 Conflict: the job exists, but it has no timeline, at least not yet.
        raise HTTPException(409, f"job {job_id} is {status.name}, not COMPLETED")
    return await handle.result()


async def get_status(handle: WorkflowHandle) -> WorkflowExecutionStatus:
    """Ask Temporal where the workflow stands. An unknown job id becomes a 404."""
    try:
        description = await handle.describe()
    except RPCError as error:
        if error.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(404, f"no job {handle.id}")
        raise
    return description.status
