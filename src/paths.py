"""Repository root resolution for config paths and artifact layout."""

from pathlib import Path

# deception/src/paths.py -> parents[1] == deception/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
