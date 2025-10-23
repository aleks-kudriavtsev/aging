"""Generate final CSV tables using existing pipeline artefacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

if __package__ is None:  # pragma: no cover - convenience for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from scripts import collect_theories, run_pipeline
from scripts.run_full_cycle import (
    _prepare_collector_config,
    _run_collector,
    _targets_from_ontology,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    default_args = run_pipeline.parse_args([])
    parser = argparse.ArgumentParser(
        description=(
            "Hydrate the collector configuration with ontology targets and "
            "export the final CSV tables."
        )
    )
    parser.add_argument(
        "--workdir",
        default="data/pipeline",
        help="Directory containing the ontology and destination for generated tables.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Collector configuration file to hydrate with ontology-derived targets.",
    )
    parser.add_argument(
        "--collector-query",
        default=default_args.collector_query,
        help="Base query string forwarded to the collector stage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional global paper export limit applied during collection.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Restrict retrieval to the specified provider names.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Override the retrieval state directory (defaults to <workdir>/collector_state).",
    )
    parser.add_argument(
        "--default-target",
        type=int,
        default=6,
        help=(
            "Optional per-node retrieval quota applied to ontology-derived targets "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard cached retrieval state before the collector run.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Alert threshold for average question confidence in the progress report.",
    )
    parser.add_argument(
        "--questions-ground-truth",
        type=Path,
        help="Optional ground-truth CSV/JSON for validating question answers after collection.",
    )
    parser.add_argument(
        "--questions-report",
        type=Path,
        help="Write the question validation report to this JSON file when provided.",
    )
    parser.add_argument(
        "--fail-on-question-mismatch",
        action="store_true",
        help=(
            "Return a non-zero exit code when question validation finds missing papers or "
            "answer mismatches."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ontology_path = workdir / "aging_ontology.json"

    try:
        targets = _targets_from_ontology(ontology_path, default_target=args.default_target)
    except FileNotFoundError:
        logger.error(
            "Ontology payload not found at %s. Run steps 1–5 of the pipeline before generating tables.",
            ontology_path,
        )
        return 1

    if not targets:
        logger.warning("No ontology groups discovered in %s; skipping collector run.", ontology_path)
        return 0

    config_path = args.config if isinstance(args.config, Path) else Path(args.config)
    config = collect_theories.load_config(config_path)
    _prepare_collector_config(config, targets=targets, ontology_path=ontology_path, workdir=workdir)

    return _run_collector(
        args=args,
        config_path=config_path,
        config=config,  # type: ignore[arg-type]
        workdir=workdir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
