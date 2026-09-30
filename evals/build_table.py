"""Build a study's results table from its runs and their checked reviews.

Usage, from the repository root, once the reviews are drafted and checked:

    uv run python evals/build_table.py runs/dev15

"Finished" counts every case of the study. The other columns use only the reviewed cases, so in a
comparison all setups are measured on the same cases. The table is printed and saved to
evals/results/<study>_table.md.
"""

import datetime as dt
import json
import sys
from pathlib import Path

from evidence_timeline.evaluation import add_scores, evaluate_case, load_reference, load_review, load_run
from evidence_timeline.models import TimelineRun

GOLD_DIR = Path("datasets/eventsmatter_v1/gold")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: build_table.py <study runs folder>")
    study_dir = Path(sys.argv[1])
    setups = sorted(path for path in study_dir.iterdir() if path.is_dir())

    rows = [
        "| Setup | Finished | Precision | Recall | Type right | Cost per case | Time per case | Output tokens per case |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for setup in setups:
        runs = [load_run(path) for path in setup.glob("case_*.json")]
        finished = sum(run.status == "completed" for run in runs)

        review_paths = sorted(Path("evals/reviews", study_dir.name, setup.name).glob("case_*.json"))
        results = []
        cost = 0.0
        seconds = 0.0
        output_tokens = 0
        for review_path in review_paths:
            run = load_run(setup / review_path.name)
            results.append(evaluate_case(run, load_reference(GOLD_DIR, run.case_id), load_review(review_path)))
            cost += run_cost(run)
            output_tokens += run_output_tokens(run)
            seconds += run_seconds(setup / "histories" / review_path.name)

        scores = add_scores([result.all_events for result in results])
        type_right = sum(result.type_right for result in results)
        cases = len(review_paths)
        rows.append(
            f"| {setup.name} | {finished}/{len(runs)} | {scores.precision:.1%} | {scores.recall:.1%} "
            f"| {type_right / scores.matched:.1%} | ${cost / cases:.3f} | {seconds / cases / 60:.1f} min "
            f"| {output_tokens / cases:,.0f} |"
        )

    table = "\n".join(rows)
    print(table)
    out_path = Path("evals/results", f"{study_dir.name}_table.md")
    out_path.write_text(table + "\n", encoding="utf-8", newline="\n")
    print(f"\nSaved to {out_path}")


def run_cost(run: TimelineRun) -> float:
    """What the provider charged for the batches and the same-event questions.

    Failed attempts are not saved in the run, so the real cost can be a little higher.
    """
    usages = [batch.usage for batch in run.batches] + [decision.usage for decision in run.pair_decisions]
    return sum((usage or {}).get("cost", 0) for usage in usages)


def run_output_tokens(run: TimelineRun) -> int:
    """Tokens the model wrote for the batches and the same-event questions, reasoning included."""
    usages = [batch.usage for batch in run.batches] + [decision.usage for decision in run.pair_decisions]
    return sum((usage or {}).get("completion_tokens", 0) for usage in usages)


def run_seconds(history_path: Path) -> float:
    """How long the case took, from the start to the end of its workflow."""
    events = json.loads(history_path.read_text(encoding="utf-8"))["events"]
    # Temporal writes nanoseconds, Python reads only microseconds, so the extra digits are cut.
    start = dt.datetime.fromisoformat(events[0]["eventTime"][:26])
    end = dt.datetime.fromisoformat(events[-1]["eventTime"][:26])
    return (end - start).total_seconds()


if __name__ == "__main__":
    main()
