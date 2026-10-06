# Clinical example dataset

Three small cases of fictional medical records, used to build and test the engine. The
[Try it](../../README.md#try-it) steps in the main README run case B.

The records are fully fictional, and no real patient data is used. They were written with AI and
checked for structure and meaning, but no clinician has reviewed them.

| Case | Documents | Events | What it tests |
|---|:---:|:---:|---|
| A | 4 | 8 | A straightforward history. |
| B | 5 | 8 | Events mentioned in more than one document. |
| C | 5 | 8 | Planned events, a date known only to the month, two documents that disagree on a date, and a document with no events. |

## Files

- `inputs/case_<X>/`: the documents, one Markdown file each. The record starts at line 8, below
  a short header.
- `domain.json` and `scope.md`: what counts as an event in this dataset. The engine reads them.
- `gold/case_<X>.json`: the answer keys, read only by the evaluator. Each event has its type,
  status, date and the exact lines it comes from. Record lines without an event are listed under
  `exclusions`, each with a reason. `gold/REVIEW_LEDGER.md` shows all 24 events in one table.
- `manifest.json`: the documents with their SHA-256 checksums, and the split.

## Checking the dataset

```bash
python validate_dataset.py
```

The script needs only Python's standard library. It checks the checksums, that every quote is on
the lines it cites, that every line of a record belongs to exactly one event or exclusion, the
date fields and the event counts. It prints a report, or stops at the first check that fails.
With `--write-report` it also saves the report to `VALIDATION_REPORT.json`.

After changing a document, update its answer key and checksum, run the check again and raise the
version in `manifest.json`.

## How it was used

Cases A and B were used during development, and C was kept for a later check. Three small cases
with public answers can show that the pipeline works, but they say little about its quality. The
records are short and tidy, so they don't test long documents, scanned pages or real clinical
writing. The evaluation in the [main README](../../README.md#results) uses court decisions
instead.
