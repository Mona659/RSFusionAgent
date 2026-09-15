"""Index lightweight summaries from persisted agent runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSummary:
    source: str
    status: str
    model: str
    experiment_mode: str
    patch_index: int | None
    metrics: dict[str, object]
    runtime_seconds: float | None
    text: str


def _summary(path: Path, payload: dict) -> ExperimentSummary:
    fusion = payload.get("fusion_result") or {}
    metrics = fusion.get("metrics") or {}
    inspection = fusion.get("inspection") or {}
    mode = fusion.get("experiment_mode") or inspection.get("experiment_mode") or "unknown"
    runtime = fusion.get("runtime") or {}
    text = (
        f"status {payload.get('status', 'unknown')}; model {payload.get('model', 'unknown')}; "
        f"mode {mode}; patch {inspection.get('patch_index')}; metrics {metrics}"
    )
    return ExperimentSummary(
        source=str(path),
        status=str(payload.get("status", "unknown")),
        model=str(payload.get("model", "unknown")),
        experiment_mode=str(mode),
        patch_index=inspection.get("patch_index"),
        metrics=metrics,
        runtime_seconds=runtime.get("runtime_seconds"),
        text=text,
    )


def load_experiment_history(root: str | Path) -> list[ExperimentSummary]:
    directory = Path(root).expanduser().resolve()
    if not directory.exists():
        return []
    results: list[ExperimentSummary] = []
    for path in sorted(directory.rglob("agent_result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append(_summary(path, payload))
    return results


def search_experiments(
    root: str | Path, query: str, *, top_k: int = 5
) -> list[ExperimentSummary]:
    if not query.strip():
        raise ValueError("Experiment query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    tokens = set(query.lower().split())
    ranked = []
    for item in load_experiment_history(root):
        score = sum(token in item.text.lower() for token in tokens)
        if score:
            ranked.append((score, item))
    return [item for _, item in sorted(ranked, key=lambda pair: (-pair[0], pair[1].source))[:top_k]]
