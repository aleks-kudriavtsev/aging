import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline.json"
    seed_papers_path = tmp_path / "seed_papers.json"
    seed_papers_path.write_text(json.dumps([]), encoding="utf-8")
    payload: Dict[str, Any] = {
        "data_sources": {
            "seed_papers": str(seed_papers_path),
        },
        "providers": [
            {
                "name": "pubmed",
                "type": "pubmed",
                "enabled": False,
            }
        ],
        "outputs": {},
        "corpus": {},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


@pytest.fixture
def write_ontology() -> Any:
    def _write_ontology(workdir: Path) -> None:
        payload = {
            "ontology": {
                "final": {
                    "groups": [
                        {
                            "name": "Cellular Mechanisms",
                            "suggested_queries": ["cellular aging mechanisms"],
                            "theories": [
                                {
                                    "preferred_label": "Senescence Cascade",
                                    "suggested_queries": ["senescence cascade aging"],
                                    "representative_titles": ["Title A"],
                                },
                                {
                                    "preferred_label": "Telomere Attrition",
                                },
                            ],
                        }
                    ]
                }
            }
        }
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "aging_ontology.json").write_text(json.dumps(payload), encoding="utf-8")

    return _write_ontology
