"""Offline trainer module for offline RL."""
from .DiscreteOfflineTrainer import (
    OfflineTrainer,
    FitConfig,
    evaluate
)

__all__ = [
    'OfflineTrainer',
    'FitConfig',
    'evaluate'
]
