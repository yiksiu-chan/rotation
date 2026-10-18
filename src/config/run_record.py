"""
Persist a YAML record of an invocation: resolved :class:`ExperimentConfig`
plus invocation metadata.

The dumped file is a drop-in ``--config``: the resolved fields live at the
top level (flat, just like ``config/default.yaml`` after section-flatten)
and the metadata sits under ``_comment``, which
:meth:`ExperimentConfig.from_dict` already pops on load.

Each call writes a new timestamped file (``run_config_<stage>_<ts>.yaml``);
nothing is overwritten. ``out_dir`` is created if missing.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.config.schema import ExperimentConfig


def _to_yaml_safe(obj: Any) -> Any:
    """Coerce ``Path`` to ``str`` recursively so PyYAML can dump the value."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_yaml_safe(v) for v in obj]
    return obj


def dump_run_record(
    cfg: ExperimentConfig,
    *,
    out_dir: Path,
    argv: Sequence[str],
    stage: str,
    steps: Sequence[str] | None = None,
) -> Path:
    """
    Write a YAML snapshot of how this run was invoked.

    Args:
        cfg: Resolved experiment config (post YAML merge + CLI overrides).
        out_dir: Directory to write into; created if missing.
        argv: Raw ``sys.argv`` for this process.
        stage: One of ``"pipeline"``, ``"behavior"``, ``"cache"``,
            ``"analysis"``, ``"plots"``. Recorded under ``_comment.stage``
            and embedded in the filename.
        steps: For ``stage == "pipeline"``, the parsed ordered step list
            (e.g. ``["behavior", "cache"]``). Ignored otherwise.

    Returns:
        Path to the written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    resolved = _to_yaml_safe(dataclasses.asdict(cfg))

    meta: dict[str, Any] = {
        "stage": stage,
        "timestamp": timestamp,
        "argv": list(argv),
    }
    if stage == "pipeline" and steps is not None:
        meta["steps"] = list(steps)

    payload: dict[str, Any] = {"_comment": meta, **resolved}

    path = out_dir / f"run_config_{stage}_{timestamp}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote run record: {path}")
    return path
