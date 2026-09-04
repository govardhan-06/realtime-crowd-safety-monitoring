from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .config import PipelineConfig
from .artifacts import config_hash, resolved_config
from .fusion import FUSION_STRATEGIES, FUSION_VERSION, build_fusion_points
from .incidents import IncidentReplay, replay_incidents
from .types import CrowdFeatureRecord, ViolenceEvidence


STRATEGIES = FUSION_STRATEGIES


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _signals(run_directory: Path) -> tuple[list[CrowdFeatureRecord], list[ViolenceEvidence]]:
    features: list[CrowdFeatureRecord] = []
    for row in _rows(run_directory / "features.jsonl"):
        features.extend(CrowdFeatureRecord(**value) for value in row["features"])
    evidence = [
        ViolenceEvidence(**{
            **row["evidence"],
            "label_mapping": tuple(tuple(item) for item in row["evidence"]["label_mapping"]),
        })
        for row in _rows(run_directory / "violence.jsonl")
    ] if (run_directory / "violence.jsonl").exists() else []
    return features, evidence


def replay_run(
    run_directory: str | Path,
    config: PipelineConfig,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, IncidentReplay]:
    run_directory = Path(run_directory)
    stored_config = json.loads((run_directory / "config.json").read_text())
    stored_values = stored_config.get("config", {})
    replay_values = resolved_config(config, Path(stored_values["input_path"]))
    replay_hash = config_hash(replay_values)
    if replay_hash != stored_config.get("config_hash"):
        raise ValueError("replay config does not match the run's resolved config hash")
    features, evidence = _signals(run_directory)
    results: dict[str, IncidentReplay] = {}
    replay_directory = run_directory / "replay"
    replay_directory.mkdir(exist_ok=True)
    (replay_directory / "metadata.json").write_text(json.dumps({
        "source_config_hash": stored_config["config_hash"],
        "replay_config_hash": replay_hash,
        "fusion_version": FUSION_VERSION,
        "strategies": list(strategies),
    }, indent=2, sort_keys=True) + "\n")
    for strategy in strategies:
        points = build_fusion_points(features, evidence, config.fusion, strategy=strategy)
        result = replay_incidents(points, config.fusion)
        results[strategy] = result
        strategy_directory = replay_directory / strategy
        strategy_directory.mkdir(exist_ok=True)
        (strategy_directory / "fusion.jsonl").write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in points))
        (strategy_directory / "incidents.jsonl").write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in result.incidents))
        (strategy_directory / "transitions.jsonl").write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in result.transitions))
    return results
