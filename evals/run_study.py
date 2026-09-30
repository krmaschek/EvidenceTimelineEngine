"""Run a study through Temporal: every case of a study file with every model and batch size.

Usage, from the repository root, with Temporal running and no other worker:

    uv run --env-file .env python evals/run_study.py evals/studies/dev15.toml 2>&1 | tee -a runs/dev15/output.txt

For each model the script starts its own worker, runs the cases one at a time and saves each run
with its Temporal history under runs/<study name>/. Runs that already exist are skipped, so after
a crash you just start it again.

The study stops at its spending limit, or after two runs in a row in which every batch failed.
One such run can be the model; two in a row usually mean the setup or an outage. To run a case
again, delete its run file and its workflow:

    docker exec evidence-timeline-temporal temporal workflow delete --workflow-id <id>
"""

import asyncio
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from temporalio.client import Client, WorkflowFailureError, WorkflowHandle, WorkflowHistory
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError

from evidence_timeline.evaluation import load_run
from evidence_timeline.models import CaseRequest, TimelineRun
from evidence_timeline.worker import DEFAULT_ADDRESS, TASK_QUEUE, connect
from evidence_timeline.workflows import BuildTimelineWorkflow

WHOLE_DOCUMENT = 10_000_000  # longer than any document, so each document is one batch


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: run_study.py <study file>")
    study = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    if not os.environ.get("LLM_API_KEY"):
        sys.exit("set LLM_API_KEY first")
    missing = [case for case in study["cases"] if not case_dir(study, case).is_dir()]
    if missing:
        sys.exit(f"case folders not found: {', '.join(missing)}")

    asyncio.run(run_study(study))


async def run_study(study: dict) -> None:
    client = await connect(DEFAULT_ADDRESS)
    limit = study["spending_limit_usd"]
    total = len(study["models"]) * len(study["batch_sizes"]) * len(study["cases"])
    # Runs saved before a restart count toward the limit too.
    saved = Path("runs", study["name"]).glob("*/case_*.json")
    spent = sum(run_cost(load_run(path)) for path in saved)
    position = 0
    failed_in_a_row = 0

    for model in study["models"]:
        settings = [(size, case) for size in study["batch_sizes"] for case in study["cases"]]
        todo = [(size, case) for size, case in settings if not run_path(study, model, size, case).exists()]
        position += len(settings) - len(todo)
        if not todo:
            continue

        worker = await start_worker(model, study)
        try:
            for size, case in todo:
                position += 1
                if spent >= limit:
                    print(f"Stopped: ${spent:.2f} spent, the limit is ${limit}.")
                    return

                label = f"[{position}/{total}] {short_name(model)} {size} {case}"
                request = CaseRequest(case_dir=str(case_dir(study, case)), max_chars=max_chars(size))
                workflow_id = f"{study['name']}-{short_name(model)}-{size}-{case}"
                try:
                    run, history, seconds = await run_case(client, workflow_id, request, worker)
                except WorkflowFailureError as error:
                    print(f"{label}: FAILED ({error.cause}); it is tried again next time")
                    continue
                if run.extractor.model != model:
                    raise RuntimeError(f"{label} ran with {run.extractor.model}: is another worker running?")

                save(run_path(study, model, size, case), run, history)
                spent += run_cost(run)
                print(
                    f"{label}: {run.status}, {int(seconds // 60)}m{int(seconds % 60):02d}s, "
                    f"${run_cost(run):.2f} (total ${spent:.2f} of ${limit})",
                    flush=True,
                )
                if run.status == "failed":
                    failed_in_a_row += 1
                else:
                    failed_in_a_row = 0
                if failed_in_a_row == 2:
                    print(f"Stopped: two runs in a row failed completely ({run.batches[0].error}). Check the cause.")
                    return
        finally:
            worker.terminate()
            worker.wait()

    print(f"Done: runs/{study['name']}/, ${spent:.2f} spent.")


async def run_case(
    client: Client, workflow_id: str, request: CaseRequest, worker: subprocess.Popen
) -> tuple[TimelineRun, WorkflowHistory, float]:
    """Run one case. Returns the run, its history and how long it took by Temporal's clock."""
    try:
        handle = await client.start_workflow(
            BuildTimelineWorkflow.run,
            request,
            id=workflow_id,
            task_queue=TASK_QUEUE,
            # A finished case is never paid for twice; a failed one can start again.
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
        )
    except WorkflowAlreadyStartedError:
        # Started before a restart: wait for it instead of paying again.
        handle = client.get_workflow_handle_for(BuildTimelineWorkflow.run, workflow_id)

    run = await wait_for_run(handle, worker)
    history = await handle.fetch_history()
    description = await handle.describe()
    return run, history, (description.close_time - description.start_time).total_seconds()


async def wait_for_run(handle: WorkflowHandle, worker: subprocess.Popen) -> TimelineRun:
    """Wait for the workflow, but give up if the worker process dies."""
    while True:
        result = asyncio.ensure_future(handle.result())
        while not result.done():
            await asyncio.wait([result], timeout=30)
            if worker.poll() is not None and not result.done():
                result.cancel()
                raise RuntimeError("the worker stopped; see its log in the study's runs folder")
        try:
            return result.result()
        except RPCError as error:
            # Only the connection dropped, e.g. while the laptop slept. The workflow is still running.
            print(f"  lost contact with Temporal ({error}); asking again", flush=True)
            await asyncio.sleep(5)


async def start_worker(model: str, study: dict) -> subprocess.Popen:
    """Start a worker for this model and wait until it is ready.

    Waiting keeps the start-up out of the first case's time. The worker's output goes to a log file.
    """
    log_path = Path("runs", study["name"], f"worker_{short_name(model)}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "evidence_timeline.cli", "worker",
        "--extractor", "llm", "--max-concurrent", str(study["max_concurrent"]),
    ]
    provider = study.get("providers", {}).get(model, "")  # empty: OpenRouter picks the host
    environment = {**os.environ, "LLM_MODEL": model, "LLM_PROVIDER": provider, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)

    for _ in range(60):
        if "Worker ready" in log_path.read_text(encoding="utf-8"):
            return worker
        if worker.poll() is not None:
            raise RuntimeError(f"the worker did not start; see {log_path}")
        await asyncio.sleep(1)
    worker.terminate()
    raise RuntimeError(f"the worker was not ready after a minute; see {log_path}")


def save(path: Path, run: TimelineRun, history: WorkflowHistory) -> None:
    """Save the history first. The run file marks the case as done, so it is written last."""
    history_path = path.parent / "histories" / path.name
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(history.to_json(), encoding="utf-8", newline="\n")
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8", newline="\n")


def run_cost(run: TimelineRun) -> float:
    """What the provider charged for the batches and the same-event questions.

    Failed attempts are not saved in the run, so the real cost can be a little higher.
    """
    usages = [batch.usage for batch in run.batches] + [decision.usage for decision in run.pair_decisions]
    return sum((usage or {}).get("cost", 0) for usage in usages)


def case_dir(study: dict, case: str) -> Path:
    return Path(study["dataset"]) / "inputs" / f"case_{case}"


def run_path(study: dict, model: str, size: int | str, case: str) -> Path:
    return Path("runs", study["name"], f"{short_name(model)}_{size}", f"case_{case}.json")


def max_chars(size: int | str) -> int:
    return WHOLE_DOCUMENT if size == "whole" else size


def short_name(model: str) -> str:
    return model.split("/")[-1]  # "google/gemini-3.8-flash" -> "gemini-3.8-flash"


if __name__ == "__main__":
    main()
