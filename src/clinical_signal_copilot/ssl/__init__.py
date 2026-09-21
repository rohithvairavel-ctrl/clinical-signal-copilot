"""Self-supervised contrastive pretraining (SimCLR / CLOCS-style)."""

from .views import augment_pair
from .encoder import TemporalEncoder, flatten_window, featurize_windows
from .contrastive import ContrastiveConfig, ContrastivePretrainer

__all__ = [
    "augment_pair",
    "TemporalEncoder",
    "flatten_window",
    "featurize_windows",
    "ContrastiveConfig",
    "ContrastivePretrainer",
]
