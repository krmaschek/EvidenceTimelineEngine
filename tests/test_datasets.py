"""A case is run with the definition of an event from its own dataset folder."""

import asyncio
import json
from pathlib import Path

from evidence_timeline.datasets import dataset_dir, load_domain
from evidence_timeline.extractors import FakeExtractor
from evidence_timeline.pipeline import run_case

COURT_DOMAIN = {
    "name": "court",
    "event_types": [{"name": "hearing", "description": "court hearings"}],
    "identifiers": "case numbers",
}


def make_court_dataset(folder: Path) -> Path:
    """A tiny dataset with one case and one document. Returns the case folder."""
    (folder / "domain.json").write_text(json.dumps(COURT_DOMAIN), encoding="utf-8")
    (folder / "scope.md").write_text("Extract hearings.\n", encoding="utf-8")
    case_dir = folder / "inputs" / "case_X"
    case_dir.mkdir(parents=True)
    (case_dir / "X01.md").write_text("The hearing took place on 3 May 2021.\n", encoding="utf-8")
    return case_dir


def test_a_case_folder_belongs_to_the_dataset_two_levels_up():
    assert dataset_dir(Path("corpus/inputs/case_X")) == Path("corpus")


def test_the_domain_is_read_from_the_dataset_folder(tmp_path):
    case_dir = make_court_dataset(tmp_path)

    domain = load_domain(case_dir)

    assert domain.name == "court"
    assert domain.scope == "Extract hearings."  # the file's final line break is not part of the scope
    assert [event_type.name for event_type in domain.event_types] == ["hearing"]


def test_a_case_is_run_with_its_own_datasets_domain(tmp_path):
    case_dir = make_court_dataset(tmp_path)

    run = asyncio.run(run_case(case_dir, FakeExtractor(), max_chars=20_000))

    assert run.domain == "court"
    assert [event.event_type for event in run.events] == ["hearing"]
