"""Command line: `extract`, `submit`, `worker`, `review-template` and `evaluate`.

`extract` runs a case in this process. `submit` hands the same work to a Temporal
worker, which `worker` starts.

Exit code 0 means success. Exit code 1 means an error or an incomplete run.
"""

import argparse
import asyncio
import datetime as dt
import os
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr

from evidence_timeline.evaluation import (
    create_review_template,
    evaluate_case,
    format_results,
    load_reference,
    load_review,
    load_run,
)
from evidence_timeline.extractors import EventExtractor, FakeExtractor
from evidence_timeline.models import CaseRequest, TimelineRun
from evidence_timeline.llm_extractor import DEFAULT_BASE_URL, LLMConfig, LLMExtractor
from evidence_timeline.llm_matcher import LLMMatcher
from evidence_timeline.pipeline import run_case
from evidence_timeline.storage import save_run
from evidence_timeline.worker import DEFAULT_ADDRESS, TASK_QUEUE, connect, run_worker
from evidence_timeline.workflows import BATCH_TIMEOUT, MATCH_TIMEOUT, BuildTimelineWorkflow


def extract(args: argparse.Namespace) -> int:
    run = asyncio.run(run_extraction(args))
    output = args.output
    if output is None:
        output = Path("runs") / f"case_{run.case_id}_{run.extractor.kind}_{run.created_at:%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    print_summary(run, output)

    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if database_url:
        asyncio.run(save_run(run, database_url))
        print(f"  saved to the database as run {run.run_id}")

    return 0 if run.status == "completed" else 1


async def run_extraction(args: argparse.Namespace) -> TimelineRun:
    if args.extractor == "fake":
        return await run_case(args.case_dir, FakeExtractor(), args.max_chars)

    config = create_llm_config(args.timeout, args.max_attempts, args.max_concurrent)
    # A "same event?" question gets the same time limit as in the Temporal worker.
    match_config = create_llm_config(MATCH_TIMEOUT.total_seconds(), args.max_attempts, args.max_concurrent)
    extractor, matcher = LLMExtractor(config), LLMMatcher(match_config)
    try:
        return await run_case(args.case_dir, extractor, args.max_chars, matcher)
    finally:
        # Close the HTTP connections even if the run crashed.
        await extractor.close()
        await matcher.close()


def create_llm_config(timeout: float, max_attempts: int, max_concurrent: int) -> LLMConfig:
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not model:
        raise ValueError("set LLM_API_KEY and LLM_MODEL (see .env.example)")
    return LLMConfig(
        base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
        api_key=SecretStr(api_key),
        model=model,
        timeout_seconds=timeout,
        max_attempts=max_attempts,
        max_concurrent_requests=max_concurrent,
        provider=os.environ.get("LLM_PROVIDER") or None,
    )


def print_summary(run: TimelineRun, output: Path) -> None:
    failed = [batch for batch in run.batches if batch.status == "failed"]
    print(f"Case {run.case_id}: {run.status.upper()}")
    if run.extractor.kind == "fake":
        print("  fake extractor: pipeline test only, not a real extraction")
    else:
        reported = sorted({batch.model for batch in run.batches if batch.model})
        print(f"  model requested: {run.extractor.model}, reported by the provider: {', '.join(reported) or '-'}")
    print(f"  batches: {len(run.batches) - len(failed)} succeeded, {len(failed)} failed")
    print(f"  lines in successful batches: {run.lines_in_successful_batches} of {run.total_lines}")
    merged = sum(bool(event.merged_from) for event in run.events)
    print(
        f"  events: {len(run.events)}, merged from several records: {merged}, "
        f"needing review: {sum(event.needs_review for event in run.events)}"
    )
    for batch in failed:
        print(f"  FAILED {batch.batch_id} ({', '.join(batch.lines)}): {batch.error}")
    print(f"  saved to {output}")


def worker(args: argparse.Namespace) -> int:
    matcher = None
    if args.extractor == "fake":
        extractor: EventExtractor = FakeExtractor()
    else:
        # Temporal owns the retries and the clock here, so the extractor tries once and waits long
        # enough that Temporal's timeout is always the one that fires.
        config = create_llm_config(BATCH_TIMEOUT.total_seconds() * 2, 1, args.max_concurrent)
        extractor, matcher = LLMExtractor(config), LLMMatcher(config)

    database_url = args.database_url or os.environ.get("DATABASE_URL")
    try:
        asyncio.run(run_worker(extractor, matcher, database_url, args.address, args.max_concurrent))
    except KeyboardInterrupt:
        print("\nWorker stopped.")
    return 0


def submit(args: argparse.Namespace) -> int:
    run = asyncio.run(submit_case(args))
    output = args.output
    if output is None:
        output = Path("runs") / f"case_{run.case_id}_{run.extractor.kind}_{run.created_at:%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    print_summary(run, output)
    return 0 if run.status == "completed" else 1


async def submit_case(args: argparse.Namespace) -> TimelineRun:
    """Start the workflow and wait. The work happens in the worker, not here."""
    client = await connect(args.address)
    request = CaseRequest(case_dir=str(args.case_dir), max_chars=args.max_chars)
    handle = await client.start_workflow(
        BuildTimelineWorkflow.run,
        request,
        id=f"timeline-{args.case_dir.name}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}",
        task_queue=TASK_QUEUE,
    )
    print(f"Workflow {handle.id} started; follow it at http://localhost:8233")
    return await handle.result()


def review_template(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"{args.output} already exists; delete it first if you really want a new review")
    run = load_run(args.run)
    reference = load_reference(args.dataset / "gold", run.case_id)
    review = create_review_template(run, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(review.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}. Fill in 'reviewer' and, for every decision, 'match' or 'reason'.")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    if len(args.run) != len(args.review):
        raise ValueError("give one --review for each --run, in the same order")
    results = []
    for run_path, review_path in zip(args.run, args.review):
        run = load_run(run_path)
        if run.extractor.kind == "fake" and not args.allow_fake:
            raise ValueError(f"{run_path} is a fake-extractor run; use --allow-fake to test the evaluator with it")
        reference = load_reference(args.dataset / "gold", run.case_id)
        results.append(evaluate_case(run, reference, load_review(review_path)))
    print(format_results(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidence-timeline")
    commands = parser.add_subparsers(required=True)

    extract_parser = commands.add_parser("extract", help="build the timeline for one case")
    extract_parser.add_argument("case_dir", type=Path, help="folder with the case's .md files")
    extract_parser.add_argument("--extractor", choices=["fake", "llm"], required=True)
    extract_parser.add_argument("--output", type=Path, help="default: runs/case_<id>_<extractor>_<time>.json")
    extract_parser.add_argument("--max-chars", type=int, default=5_000, help="characters per batch")
    extract_parser.add_argument(
        "--timeout", type=float, default=BATCH_TIMEOUT.total_seconds(), help="seconds allowed per batch request"
    )
    extract_parser.add_argument("--max-attempts", type=int, default=3, help="tries per batch")
    extract_parser.add_argument("--max-concurrent", type=int, default=4, help="LLM requests running at the same time")
    extract_parser.add_argument("--database-url", help="default: DATABASE_URL; without it the run is only saved as JSON")
    extract_parser.set_defaults(handler=extract)

    worker_parser = commands.add_parser("worker", help="run a Temporal worker that carries out cases")
    worker_parser.add_argument("--extractor", choices=["fake", "llm"], required=True)
    worker_parser.add_argument("--max-concurrent", type=int, default=4, help="activities running at the same time")
    worker_parser.add_argument("--address", default=DEFAULT_ADDRESS, help="Temporal server")
    worker_parser.add_argument("--database-url", help="default: DATABASE_URL; without it runs are not saved")
    worker_parser.set_defaults(handler=worker)

    submit_parser = commands.add_parser("submit", help="give one case to a worker and wait for the timeline")
    submit_parser.add_argument("case_dir", type=Path, help="folder with the case's .md files")
    submit_parser.add_argument("--max-chars", type=int, default=5_000, help="characters per batch")
    submit_parser.add_argument("--address", default=DEFAULT_ADDRESS, help="Temporal server")
    submit_parser.add_argument("--output", type=Path, help="default: runs/case_<id>_<extractor>_<time>.json")
    submit_parser.set_defaults(handler=submit)

    template_parser = commands.add_parser("review-template", help="write an empty review file for a run")
    template_parser.add_argument("run", type=Path)
    template_parser.add_argument("--dataset", type=Path, required=True, help="dataset folder with the gold answers")
    template_parser.add_argument("--output", type=Path, required=True)
    template_parser.set_defaults(handler=review_template)

    evaluate_parser = commands.add_parser("evaluate", help="score runs using their review files")
    evaluate_parser.add_argument("--dataset", type=Path, required=True, help="dataset folder with the gold answers")
    evaluate_parser.add_argument("--run", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--review", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--allow-fake", action="store_true")
    evaluate_parser.set_defaults(handler=evaluate)

    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
