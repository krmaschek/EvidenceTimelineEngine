# Evidence Timeline Engine: starter dataset v1.0.0

An original, fully fictional English-language dataset for developing an evidence-linked timeline extractor. No real patient data and no Synthea data are used. Clinical statements are test fixtures, not medical guidance or evidence of clinical realism. The dataset was AI-authored and checked structurally and by semantic review; it has not been independently reviewed by a clinician.

## Contents and use

- `inputs/case_A/`: 4 Markdown documents; straightforward history; 8 events.
- `inputs/case_B/`: 5 Markdown documents; repeated mentions; 8 events.
- `inputs/case_C/`: 5 Markdown documents; planned events, approximate and conflicting dates, and one administrative-only document; 8 events.
- `gold/`: evaluator-only JSON references and a review ledger.
- `manifest.json`: document inventory, input checksums and proposed development/holdout split.
- `validate_dataset.py`: offline standard-library validation; no model or API key required.
- `VALIDATION_REPORT.json`: result of the shipped validation run.

Run `python validate_dataset.py` from this directory (or invoke the script by absolute path). It prints a JSON report and exits nonzero on failure. Run with `--write-report` to refresh the saved report.

Give the extractor only one case's `inputs/case_X/*.md` and the event-scope instructions below. Never supply `gold/`, the review ledger or the validation script as model evidence. Keep patient cases separate. Document IDs and line numbers are provenance identifiers, not event labels. These are Markdown inputs: cite lines, not invented page numbers.

Use A and B during development. Reserve C for a later check without tuning on its documents or labels. Because the challenge categories and references are shipped openly, C is a convenience holdout, not an independent blind benchmark. Do not claim statistical generalization from three cases. Add new independently authored cases before making broader quality claims.

## Extraction scope (supply this section to the extractor)

Extract explicitly documented visits/assessments, diagnostic procedures, supervised exercise therapy sessions, and actual medication starts. Also extract definite scheduled procedures, with `status=planned`. Completed means the source reports that the event occurred; a patient's report may support an event, but is not independent clinical verification.

Do not infer an additional visit merely because a procedure took place. When a visit and a procedure are both explicitly documented, they are two events, even on the same date. Distinct therapy sessions on different dates are separate events. Repeated descriptions of the same event are one event with multiple supporting sources. Identical dates alone do not establish duplication.

Exclude symptoms alone, negated events, general advice, administrative updates, document authoring dates, and hypothetical/conditional treatments that were not definitely scheduled. A mention of an already-started medicine does not establish another medication start. Preserve the difference between planned and completed events. The passage of a scheduled date does not prove that treatment occurred.

For an exact date, use ISO YYYY-MM-DD. For an approximate date, preserve its original wording, leave the precise `date` null, and record only the supported interval. If sources explicitly disagree about the same event's date and neither is verified, leave `date` null, preserve the alternatives and both sources, and flag review. Do not resolve a conflict merely by selecting the later-authored document.

Order events using the dates supported by evidence. Do not invent ordering within a day. Approximate events may be positioned relative to non-overlapping dates, but their displayed precision must stay approximate. Retain unresolved events rather than dropping them. Record exact source quotations and source locations for each event. Do not emit a numerical confidence score.

## Gold schema and annotation conventions

Each case JSON contains `events`, `exclusions`, and `zero_event_documents`. All 24 event IDs are evaluation identifiers; the extractor is not expected to reproduce them.

- `event_type`: `visit`, `procedure`, or `medication_start`.
- `status`: `completed` or `planned`.
- `date`: exact event date, or null. For planned events, this is the scheduled date, not the scheduling action's date.
- `date_precision`: `day`, `month`, or `conflicting`.
- `date_interval`: inclusive earliest/latest bounds for approximate dates, otherwise null.
- `alternative_dates`: explicit competing dates for one event, not a continuous date range.
- `sources`: document ID, inclusive 1-based line range, exact quote, and original event date wording (`date_text`). No PDF page fields apply.
- `needs_review` and `review_reasons`: the expected flags for approximate/conflicting dates in this dataset. They are not model confidence estimates.

Every narrative line is assigned either to an event's evidence or to an explicit exclusion. Titles, synthetic-data banners, case/document IDs and record-created metadata are not clinical event statements. Exact quotations are deliberately sentence-sized. For C-E06, each alternative date is linked to its own source through `date_text`; both reports concern CU-77.

Descriptions are reference paraphrases, not strings requiring exact output equality. The JSON reference schema can be adapted to the application, provided the evidence, uncertainty and status distinctions remain intact.

## Evaluation protocol

Evaluate two views:

1. All in-scope events, including explicitly planned procedures (24 reference events).
2. Completed events only (22 reference events). Do not count plans as completed care.

For this tiny set, begin with a human-reviewed one-to-one matching table between predictions and references. Match case, event type, action/body site or procedure identifier, and status. For exact events require the correct date; for uncertain events require the supported interval or explicit conflict, with no invented precise date. Allow equivalent wording. One prediction can match at most one reference, and one reference at most one prediction. Extra duplicates count as false positives; missing references count as false negatives. Record mismatch reasons to distinguish date/status errors from omissions. Do not silently score mock outputs as LLM quality.

Report precision = matched predictions / all predictions and recall = matched references / all reference events. Report N/A precision when there are no predictions; for a zero-event document report the false-positive count rather than undefined recall. Aggregate case counts for micro scores; show case-level results too.

Score provenance separately: a valid quote must occur in the stated document/lines AND support the claimed event, date and status. Report evidence-location validity and semantic support separately. Report whether all annotated repeated mentions were retained; a correctly extracted event with one source may pass event matching while still losing provenance. There are 6 events with multiple document mentions, containing 13 mentions total; 7 mentions are redundant beyond one per event.

Report whether the one date conflict and one approximate-date case were correctly flagged; show raw counts. Check the administrative-only C05 for zero event predictions. The duplicated MRI booking is still planned in both sources. C05 is dated on the scheduled X-ray day but contains no evidence that the X-ray occurred.

No performance metrics are prefilled. `VALIDATION_REPORT.json` describes fixture integrity, not extractor accuracy. This package intentionally supplies a validation script, not an automatic semantic event matcher or a trained extraction model.

## Limits and next step

These short, controlled records test scope, event identity, source fidelity and elementary date handling. They do not test OCR, page boundaries, long contexts, clinical reasoning or the full diversity of real records. Start Phase 1 on these files before expanding the corpus or generating PDFs. Maintain the answer key when changing a document; rerun validation and increment the dataset version.

The proposed application schema must retain event status and date uncertainty. A timeline that stores only an exact date and a description cannot represent all of this dataset correctly.
