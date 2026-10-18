"""Task-specific corpora loaders (TriviaQA JSON, Azaria HF).

Named ``task_corpus`` to avoid clashing with the Hugging Face ``datasets`` package.

``load_azaria_records`` is imported from ``src.task_corpus.azaria_loader`` directly so
importing this package does not require the Hugging Face ``datasets`` dependency.
"""

from src.task_corpus.records import StatementRecord
from src.task_corpus.triviaqa_loader import load_triviaqa_filtered_records

__all__ = ["StatementRecord", "load_triviaqa_filtered_records"]
