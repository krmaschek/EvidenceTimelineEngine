"""Command line: `extract`, `review-template` and `evaluate`.

Exit code 0 means success. Exit code 1 means an error or an incomplete run.
"""

import argparse
import asyncio
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
from evidence_timeline.extractors import FakeExtractor
from evidence_timeline.models import TimelineRun
from evidence_timeline.llm_extractor import DEFAULT_BASE_URL, LLMConfig, LLMExtractor
from evidence_timeline.pipeline import run_case

GOLD_DIR = Path("evidence_timeline_dataset_v1/gold")


def extract(args: argparse.Namespace) -> int:
    run = asyncio.run(run_extraction(args))
    output = args.output
    if output is None:
        output = Path("runs") / f"case_{run.case_id}_{run.extractor.kind}_{run.created_at:%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    print_summary(run, output)
    return 0 if run.status == "completed" else 1


async def run_extraction(args: argparse.Namespace) -> TimelineRun:
    if args.extractor == "fake":
        return await run_case(args.case_dir, FakeExtractor(), args.max_chars)

    extractor = create_llm_extractor(args)
    try:
        return await run_case(args.case_dir, extractor, args.max_chars)
    finally:
        await extractor.close()  # close the HTTP connection even if the run crashed


def create_llm_extractor(args: argparse.Namespace) -> LLMExtractor:
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not model:
        raise ValueError("set LLM_API_KEY and LLM_MODEL (see .env.example)")
    config = LLMConfig(
        base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
        api_key=SecretStr(api_key),
        model=model,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
        max_concurrent_requests=args.max_concurrent,
    )
    return LLMExtractor(config)


def print_summary(run: TimelineRun, output: Path) -> None:
    failed = [batch for batch in run.batches if batch.status == "failed"]
    print(f"Case {run.case_id}: {run.status.upper()} (preliminary timeline, events may repeat)")
    if run.extractor.kind == "fake":
        print("  fake extractor: pipeline test only, not a real extraction")
    else:
        reported = sorted({batch.model for batch in run.batches if batch.model})
        print(f"  model requested: {run.extractor.model}, reported by the provider: {', '.join(reported) or '-'}")
    print(f"  batches: {len(run.batches) - len(failed)} succeeded, {len(failed)} failed")
    print(f"  lines in successful batches: {run.lines_in_successful_batches} of {run.total_lines}")
    print(f"  events: {len(run.events)}, needing review: {sum(event.needs_review for event in run.events)}")
    for batch in failed:
        print(f"  FAILED {batch.batch_id} ({', '.join(batch.lines)}): {batch.error}")
    print(f"  saved to {output}")


def review_template(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"{args.output} already exists; delete it first if you really want a new review")
    run = load_run(args.run)
    review = create_review_template(run, load_reference(GOLD_DIR, run.case_id))
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
        results.append(evaluate_case(run, load_reference(GOLD_DIR, run.case_id), load_review(review_path)))
    print(format_results(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidence-timeline")
    commands = parser.add_subparsers(required=True)

    extract_parser = commands.add_parser("extract", help="build a preliminary timeline for one case")
    extract_parser.add_argument("case_dir", type=Path, help="folder with the case's .md files")
    extract_parser.add_argument("--extractor", choices=["fake", "llm"], required=True)
    extract_parser.add_argument("--output", type=Path, help="default: runs/case_<id>_<extractor>_<time>.json")
    extract_parser.add_argument("--max-chars", type=int, default=20_000, help="characters per batch")
    extract_parser.add_argument("--timeout", type=float, default=120, help="seconds allowed per LLM request")
    extract_parser.add_argument("--max-attempts", type=int, default=3, help="tries per batch")
    extract_parser.add_argument("--max-concurrent", type=int, default=4, help="LLM requests running at the same time")
    extract_parser.set_defaults(handler=extract)

    template_parser = commands.add_parser("review-template", help="write an empty review file for a run")
    template_parser.add_argument("run", type=Path)
    template_parser.add_argument("--output", type=Path, required=True)
    template_parser.set_defaults(handler=review_template)

    evaluate_parser = commands.add_parser("evaluate", help="score runs using their review files")
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
