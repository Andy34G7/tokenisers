"""
tokenisers: Modular Rust & Python Research and Benchmarking Platform for Tokenizers.

``data``, ``benchmarks`` and ``DynamicResearchTokenizer`` (torch/transformers) load on first use;
they are left out of ``__all__`` so ``from tokenisers import *`` stays light.
"""

import importlib

from tokenisers import _core
from tokenisers.models import (
    BpeTokenizer,
    ByteEntropyTokenizer,
    LengthPenalizedTokenizer,
    MdlTokenizer,
    MorphemeBpeTokenizer,
    MorphemeGuidedTokenizer,
    SuperBpeTokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
    WpBpeBaseTokenizer,
    WpBpeFtTokenizer,
    WpBpePigTokenizer,
    WpBpeScpTokenizer,
    WpBpeNpmiTokenizer,
    WpBpePpigTokenizer,
)

__version__ = "0.1.0"

_LAZY = {
    "data": ("tokenisers.data", None),
    "benchmarks": ("tokenisers.benchmarks", None),
    "DynamicResearchTokenizer": ("tokenisers.wrapper", "DynamicResearchTokenizer"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'tokenisers' has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "_core",
    "BpeTokenizer",
    "WordPieceTokenizer",
    "WpBpeBaseTokenizer",
    "WpBpeFtTokenizer",
    "WpBpePigTokenizer",
    "WpBpeScpTokenizer",
    "WpBpeNpmiTokenizer",
    "WpBpePpigTokenizer",
    "MdlTokenizer",
    "LengthPenalizedTokenizer",
    "SuperBpeTokenizer",
    "MorphemeGuidedTokenizer",
    "MorphemeBpeTokenizer",
    "UnigramTokenizer",
    "ByteEntropyTokenizer",
]
