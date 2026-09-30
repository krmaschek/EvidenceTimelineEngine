"""Draft a review for every run of a study, so a person only has to check it.

Usage, from the repository root:

    uv run python evals/draft_reviews.py runs/dev15

When exactly one gold event sits on the lines a prediction cites, with the same status and date,
the draft matches the two. Everything less certain is drafted too, with a note starting "check:",
so the reviewer reads it first. Every decision is checked before the review is scored.

A study of several setups drafts only the cases all of them completed, so the setups are compared
on the same cases. A study of one setup is a final test, so every case counts, even one that lost
a batch. Reviews go to evals/reviews/<study>/<setup>/ and are never overwritten.
"""

import sys
from pathlib import Path

from evidence_timeline.evaluation import (
    ReferenceCase,
    ReferenceEvent,
    Review,
    create_review_template,
    load_reference,
    load_run,
    rule_problems,
)
from evidence_timeline.models import TimelineEvent, TimelineRun

GOLD_DIR = Path("datasets/eventsmatter_v1/gold")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: draft_reviews.py <study runs folder>")
    study_dir = Path(sys.argv[1])
    setups = sorted(path for path in study_dir.iterdir() if path.is_dir())

    cases = [path.stem.removeprefix("case_") for path in setups[0].glob("case_*.json")]
    chosen = []
    for case in cases:
        statuses = [load_run(setup / f"case_{case}.json").status for setup in setups]
        # With one setup this is a final test, where every case counts.
        if len(setups) == 1 or all(status == "completed" for status in statuses):
            chosen.append(case)
    left_out = sorted(set(cases) - set(chosen))
    print(f"Drafting {len(chosen)} of {len(cases)} cases.")
    if left_out:
        print(f"Left out, not completed by every setup: {', '.join(left_out)}")

    for setup in setups:
        out_dir = Path("evals/reviews", study_dir.name, setup.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        for case in sorted(chosen):
            path = out_dir / f"case_{case}.json"
            if path.exists():
                continue
            run = load_run(setup / f"case_{case}.json")
            review = draft_review(run, load_reference(GOLD_DIR, case))
            path.write_text(review.model_dump_json(indent=2), encoding="utf-8", newline="\n")
            to_check = sum(decision.note.startswith("check:") for decision in review.decisions)
            print(f"{setup.name} {case}: {len(review.decisions)} decisions, {to_check} to check")


def draft_review(run: TimelineRun, reference: ReferenceCase) -> Review:
    review = create_review_template(run, reference)
    review.reviewer = "draft by evals/draft_reviews.py, not checked yet"
    predictions = {event.event_id: event for event in run.events}
    used: set[str] = set()  # gold events already matched in this review

    for decision in review.decisions:
        prediction = predictions[decision.prediction_id]
        on_line = gold_on_cited_lines(prediction, reference)
        fitting = [gold for gold in on_line if not rule_problems(prediction, gold)]
        free = [gold for gold in fitting if gold.event_id not in used]

        if not on_line:
            decision.reason = "not_a_reference_event"
            decision.note = "check: no gold event on the cited lines"
        elif not fitting:
            problem = rule_problems(prediction, on_line[0])[0]
            decision.reason = "wrong_status" if problem.startswith("status") else "wrong_date"
            decision.note = f"rule: {on_line[0].event_id} is on the line, but {problem}"
        elif not free:
            decision.reason = "duplicate"
            decision.note = f"check: {fitting[0].event_id} is already matched"
        else:
            decision.match = free[0].event_id
            used.add(free[0].event_id)
            if len(fitting) == 1:
                decision.note = "rule: the only gold event on the cited lines"
            else:
                decision.note = "check: several gold events on the cited lines"
    return review


def gold_on_cited_lines(prediction: TimelineEvent, reference: ReferenceCase) -> list[ReferenceEvent]:
    """The gold events that start on a line the prediction quotes."""
    found = []
    for gold in reference.events:
        for source in gold.sources:
            for quote in prediction.evidence:
                same_document = quote.document_id == source.document_id
                on_quoted_lines = quote.line_start <= source.line_start <= quote.line_end
                if same_document and on_quoted_lines and gold not in found:
                    found.append(gold)
    return found


if __name__ == "__main__":
    main()
