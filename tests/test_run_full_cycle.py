import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_full_cycle
from theories_pipeline.outputs import (
    COMPETITION_PAPER_COLUMNS,
    COMPETITION_QUESTION_COLUMNS,
    COMPETITION_THEORY_COLUMNS,
    COMPETITION_THEORY_PAPER_COLUMNS,
    QUESTION_COLUMNS,
    QUESTION_CONFIDENCE_COLUMNS,
)


def test_prepare_collector_config_rewrites_outputs(tmp_path: Path) -> None:
    workdir = tmp_path / "prepared"
    targets = {"Example": {"target": 10}}
    ontology_path = tmp_path / "aging_ontology.json"

    cli_questions = tmp_path / "cli_questions.csv"

    config: Dict[str, Any] = {
        "corpus": {},
        "outputs": {
            "papers": "data/examples/papers.csv",
            "theories": "data/examples/theories.csv",
            "theory_papers": "data/examples/theory_papers.csv",
            "questions": "data/examples/questions.csv",
            "cache_dir": "data/cache",
            "reports": "data/reports",
            "competition": {
                "papers": "data/examples/competition/papers.csv",
                "theories": "data/examples/competition/theories.csv",
                "theory_papers": "data/examples/competition/theory_papers.csv",
                "questions": "data/examples/competition/questions.csv",
            },
        },
    }
    config["outputs"]["competition"]["questions"] = cli_questions

    run_full_cycle._prepare_collector_config(
        config,
        targets=targets,
        ontology_path=ontology_path,
        workdir=workdir,
    )

    outputs_cfg = config["outputs"]
    assert outputs_cfg["papers"] == str(workdir / "papers.csv")
    assert outputs_cfg["theories"] == str(workdir / "theories.csv")
    assert outputs_cfg["theory_papers"] == str(workdir / "theory_papers.csv")
    assert outputs_cfg["questions"] == str(workdir / "questions.csv")
    assert outputs_cfg["cache_dir"] == str(workdir / "cache")
    assert outputs_cfg["reports"] == str(workdir / "reports")

    competition_cfg = outputs_cfg["competition"]
    expected_competition_dir = workdir / "competition"
    assert competition_cfg["base_dir"] == str(expected_competition_dir)
    assert competition_cfg["papers"] == str(expected_competition_dir / "papers.csv")
    assert competition_cfg["theories"] == str(expected_competition_dir / "theories.csv")
    assert competition_cfg["theory_papers"] == str(
        expected_competition_dir / "theory_papers.csv"
    )
    assert competition_cfg["questions"] == str(cli_questions)


def test_configure_offline_mode_disables_providers() -> None:
    config: Dict[str, Any] = {
        "providers": [
            {"name": "openalex", "type": "openalex", "enabled": True},
            {"name": "pubmed", "type": "pubmed", "enabled": True},
        ],
        "corpus": {
            "bootstrap": {
                "enabled": True,
                "providers": ["openalex", "pubmed"],
                "queries": {
                    "seed": {
                        "query": "aging theory",
                        "providers": ["openalex", "pubmed"],
                    }
                },
            },
            "expansion": {"enabled": True},
        },
    }

    run_full_cycle._configure_offline_mode(config)

    assert config["providers"] == []
    corpus_cfg = config["corpus"]
    bootstrap_cfg = corpus_cfg["bootstrap"]
    assert bootstrap_cfg["enabled"] is False
    assert bootstrap_cfg.get("providers") == []
    for entry in bootstrap_cfg.get("queries", {}).values():
        if isinstance(entry, Mapping):
            assert "providers" not in entry
    expansion_cfg = corpus_cfg["expansion"]
    assert expansion_cfg["enabled"] is False


def test_run_full_cycle_invokes_pipeline_and_collector(
    tmp_path: Path, tmp_config: Path, write_ontology, monkeypatch
) -> None:
    workdir = tmp_path / "cycle"

    pipeline_calls: List[List[str]] = []

    def fake_run_pipeline_main(argv: List[str] | None) -> int:
        pipeline_calls.append(list(argv or []))
        target_dir = Path(argv[argv.index("--workdir") + 1]) if argv else workdir
        write_ontology(Path(target_dir))
        return 0

    monkeypatch.setattr(run_full_cycle.run_pipeline, "main", fake_run_pipeline_main)

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
        "5",
        "--no-resume",
    ]

    result = run_full_cycle.main(args)
    assert result == 0

    assert pipeline_calls, "run_pipeline.main should be invoked"
    assert any("--workdir" in call for call in pipeline_calls)
    assert collector_calls, "collect_for_entry should be executed"

    state_dir = workdir / "collector_state"
    assert state_dir.exists(), "Collector state directory should default under the workdir"

    papers_path = workdir / "papers.csv"
    theories_path = workdir / "theories.csv"
    theory_papers_path = workdir / "theory_papers.csv"
    questions_path = workdir / "questions.csv"
    for path in [papers_path, theories_path, theory_papers_path, questions_path]:
        assert path.exists()

    with questions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "theory_id",
            "paper_url",
            "paper_name",
            "paper_year",
            *QUESTION_COLUMNS,
            *QUESTION_CONFIDENCE_COLUMNS,
        ]

    competition_dir = workdir / "competition"
    assert competition_dir.exists()
    competition_paths = {
        "papers": competition_dir / "papers.csv",
        "theories": competition_dir / "theories.csv",
        "theory_papers": competition_dir / "theory_papers.csv",
        "questions": competition_dir / "questions.csv",
    }
    for path in competition_paths.values():
        assert path.exists()

    with competition_paths["papers"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_PAPER_COLUMNS)

    with competition_paths["theories"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_COLUMNS)

    with competition_paths["theory_papers"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_PAPER_COLUMNS)

    with competition_paths["questions"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_QUESTION_COLUMNS)

    accuracy_ground_truth = [
        {
            "theory_id": "placeholder",
            "paper_id": "placeholder",
            "question_id": question,
            "expected_answer": "",
        }
        for question in QUESTION_COLUMNS
    ]
    assert len(accuracy_ground_truth) == len(QUESTION_COLUMNS)


def test_run_full_cycle_skip_pipeline_reuses_existing_ontology(
    tmp_path: Path, tmp_config: Path, write_ontology, monkeypatch
) -> None:
    workdir = tmp_path / "existing"
    write_ontology(workdir)

    def fail_run_pipeline_main(argv: List[str] | None) -> int:  # pragma: no cover - defensive
        raise AssertionError("run_pipeline.main should not be invoked when --skip-pipeline is set")

    monkeypatch.setattr(run_full_cycle.run_pipeline, "main", fail_run_pipeline_main)

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
        "--skip-pipeline",
    ]

    result = run_full_cycle.main(args)
    assert result == 0

    assert collector_calls, "Collector should be executed when reusing existing ontology"

    outputs_dir = {
        "papers": workdir / "papers.csv",
        "theories": workdir / "theories.csv",
        "theory_papers": workdir / "theory_papers.csv",
        "questions": workdir / "questions.csv",
    }
    for path in outputs_dir.values():
        assert path.exists()

    competition_dir = workdir / "competition"
    assert competition_dir.exists()
