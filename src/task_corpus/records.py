"""Normalized row types shared across loaders and prompt construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StatementRecord:
    """
    One evaluable statement with integer ground-truth label.

    ``label`` uses the project convention: 0 → GT_0 (e.g. False / Left),
    1 → GT_1 (e.g. True / Right) for the configured task.
    """

    index: int
    claim_text: str
    label: int
    pair_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
