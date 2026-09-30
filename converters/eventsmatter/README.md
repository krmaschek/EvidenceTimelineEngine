# EventsMatter converter

Turns the [EventsMatter corpus](https://zenodo.org/records/4032617) into a dataset folder that the
Evidence Timeline pipeline can read. The pipeline never imports this code; it only reads the folder
this script writes.

```bash
uv run python converters/eventsmatter/convert.py --output datasets/eventsmatter_v1
```

The script downloads the corpus (3.5 MB), checks it against the published MD5, and writes:

| Path | Contents |
|---|---|
| `inputs/case_<id>/<id>.md` | one decision, one paragraph per line |
| `gold/case_<id>.json` | the annotated events of that decision |
| `manifest.json` | the source, the splits and the counts |

These files are not committed; run the script again to recreate them.

The dataset's definition of an event, `domain.json` and `scope.md`, is not written by this script.
It is written by hand from the annotation guidelines and committed in the dataset folder itself.
The type descriptions paraphrase the guidelines' definition (section 3.1) and their rule for
difficult cases (section 3.3). Their examples are left out because they quote decisions of the
corpus, including holdout ones.

## The corpus

30 decisions of the European Court of Human Rights from HUDOC, annotated by two legal experts in
GATE and then reconciled into a consensus: 615 events, each with a type (`procedure` or
`circumstance`) and its what, who and when. Annotation guidelines:
<https://mnavasloro.github.io/EventsMatter/Guidelines.pdf>.

License: GPL-2.0. Cite: E. Filtz, M. Navas-Loro, C. Santos, A. Polleres, S. Kirrane (2020),
*Events Matter: Extraction of Events from Court Decisions*, JURIX 2020.

## Choices made in the conversion

- **Consensus annotations only.** The two annotators' own annotations are ignored.
- **One paragraph per line.** The text is taken without its markup, empty paragraphs are dropped,
  and whitespace inside a paragraph (line breaks, non-breaking spaces) becomes single spaces. A gold
  source is the line or lines of the paragraphs its event covers, with the event's text as the quote.
- **Dates.** Only a single "when" written as a day ("5 November 2010"), a month ("November 2010")
  or a year ("2010") is scored. Relative or vague wordings ("the same day", "late summer 2010") and
  events with several or no "when" parts get `date_scored: false`. About 70% of the dates are scored.
- **Description** is the event's who followed by its what, e.g. "applicant lodged two complaints".
- **Status** is always `completed`: the corpus marks events that did not happen, and none of the
  consensus events are marked that way.
- **Splits.** Their `train` decisions (24) are for development; their `dev` and `test` decisions (6)
  are the holdout.

## Known imperfections

- In two decisions (MOSKALEV, RESIN) two events share one id, so both get the same what, who and when.
- One event is marked `importance: L` (only one annotator wanted it); it is kept like the others.
- Four events have no what annotated, so their quote becomes their description. One of them
  (S_N_v_RUSSIA-E21) has no when either, so its date is unscored although the sentence gives one.
- The corpus annotates the parts of a decision that tell the case's story, not the Court's legal
  reasoning; `scope.md` asks the model to do the same.
- The guidelines (section 3.2) mark events that did not happen, such as a party not appealing, and
  leave them out of the timeline; `scope.md` asks the model to leave them out too. Some consensus
  events are exactly that but carry no mark, e.g. "local administration did not appeal"
  (YEVGENIY_ZAKHAROV_v_RUSSIA-E13) and "applicants did not lodge a cassation appeal"
  (PANYUSHKINY_v_RUSSIA-E26). A model that follows the guidelines misses them.
