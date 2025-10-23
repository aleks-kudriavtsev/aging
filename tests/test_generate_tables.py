import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_tables, run_full_cycle


def test_generate_tables_runs_collector(
    tmp_path: Path, tmp_config: Path, write_ontology, monkeypatch
) -> None:
    workdir = tmp_path / "tables"
    write_ontology(workdir)

    collector_calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    def fake_collect_for_entry(*args, **kwargs):
        collector_calls.append((args, kwargs))
        return {"total_unique": 0}, []

    monkeypatch.setattr(run_full_cycle.collect_theories, "collect_for_entry", fake_collect_for_entry)
    monkeypatch.setattr(run_full_cycle.collect_theories, "_load_api_keys", lambda *a, **k: {})
    monkeypatch.setattr(run_full_cycle.collect_theories, "_maybe_build_llm_client", lambda *a, **k: None)

    class DummyClassifier:
        def attach_manager(self, manager: Any) -> None:  # pragma: no cover - trivial
            self.manager = manager

        def summarize(self, assignments: List[Any], *, include_ids: bool = False) -> Dict[str, Any]:
            return {}

        @classmethod
        def from_config(cls, *a, **k):  # pragma: no cover - simple factory
            return cls()

    monkeypatch.setattr(run_full_cycle.collect_theories, "TheoryClassifier", DummyClassifier)
    monkeypatch.setattr(run_full_cycle.collect_theories, "QuestionExtractor", lambda *a, **k: object())
    monkeypatch.setattr(
        run_full_cycle.collect_theories,
        "classify_and_extract_parallel",
        lambda papers, classifier, extractor, workers=1: ([], []),
    )

    args = [
        "--workdir",
        str(workdir),
        "--config",
        str(tmp_config),
        "--collector-query",
        "aging theory",
        "--limit",
        "3",
    ]

    result = generate_tables.main(args)
    assert result == 0

    assert collector_calls, "Collector should be invoked to generate tables"

    outputs_dir = {
        "papers": workdir / "papers.csv",
        "theories": workdir / "theories.csv",
        "theory_papers": workdir / "theory_papers.csv",
        "questions": workdir / "questions.csv",
    }
    for path in outputs_dir.values():
        assert path.exists()

    with outputs_dir["papers"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None


def test_generate_tables_errors_without_ontology(tmp_path: Path, tmp_config: Path) -> None:
    args = [
        "--workdir",
        str(tmp_path / "missing"),
        "--config",
        str(tmp_config),
        "--collector-query",
        "aging theory",
    ]

    result = generate_tables.main(args)
    assert result == 1
