import pytest
from helpers import CASE_A_DIR, DATASET_DIR, REPO_ROOT

from evidence_timeline.cli import main
from evidence_timeline.models import TimelineRun


@pytest.fixture(autouse=True)
def repo_root_without_credentials(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # relative paths such as runs/ stay inside the repository
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)  # a test must not write into a real database


def test_extract_with_the_fake_extractor(tmp_path, capsys):
    output = tmp_path / "run.json"

    assert main(["extract", str(CASE_A_DIR), "--extractor", "fake", "--output", str(output)]) == 0

    assert TimelineRun.model_validate_json(output.read_text(encoding="utf-8")).status == "completed"
    assert "fake extractor: pipeline test only" in capsys.readouterr().out


def test_extract_with_the_llm_needs_credentials(capsys):
    assert main(["extract", str(CASE_A_DIR), "--extractor", "llm"]) == 1
    assert "set LLM_API_KEY and LLM_MODEL" in capsys.readouterr().err


def test_review_template_and_evaluate(tmp_path, capsys):
    run_file = tmp_path / "run.json"
    review_file = tmp_path / "review.json"
    main(["extract", str(CASE_A_DIR), "--extractor", "fake", "--output", str(run_file)])

    template = ["review-template", str(run_file), "--dataset", str(DATASET_DIR), "--output", str(review_file)]
    assert main(template) == 0
    # An existing review is never overwritten.
    assert main(template) == 1
    # Fake runs are not scored unless explicitly allowed.
    evaluate = ["evaluate", "--dataset", str(DATASET_DIR), "--run", str(run_file), "--review", str(review_file)]
    assert main(evaluate) == 1
    assert "fake-extractor run" in capsys.readouterr().err
