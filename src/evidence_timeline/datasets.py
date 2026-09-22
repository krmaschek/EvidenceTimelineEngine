"""Find a case's dataset and read its definition of an event.

Every dataset is a folder with the same layout:

    <dataset>/
        domain.json          the event types, and what descriptions should keep
        scope.md             the extraction scope: which events to extract
        inputs/<case>/*.md   the documents of each case
        gold/                reference answers, read only by the evaluator

So a case folder is always <dataset>/inputs/<case>/, and its dataset is two levels
up. Running another corpus means giving the pipeline a case folder from another
dataset; nothing in the code names a corpus.
"""

import json
from pathlib import Path

from evidence_timeline.models import Domain


def dataset_dir(case_dir: Path) -> Path:
    """The dataset folder of a case: <dataset>/inputs/<case>/ -> <dataset>/."""
    return case_dir.parent.parent


def load_domain(case_dir: Path) -> Domain:
    folder = dataset_dir(case_dir)
    fields = json.loads((folder / "domain.json").read_text(encoding="utf-8"))
    scope = (folder / "scope.md").read_text(encoding="utf-8").strip()
    return Domain(**fields, scope=scope)
