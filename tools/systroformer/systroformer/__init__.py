"""Systroformer: a transformer block whose FFN output is modulated by the
algebraic connectivity (λ₂) of the address-space Hamming graph of its
own attention activations.

Built on the Systrophe address-space catcher (`systrophe.novelty_catcher`).

The core idea — emergent information-topological structure detected per
forward pass — comes from the helper file `SystropheLLMhelper.txt`
(Systroformer prototype design). This is the first derived tool that
plugs a Systrophe primitive into an LLM architecture.
"""

from .catcher import (
    address_from_activation,
    derivative_catcher,
    hamming_graph_lambda2,
    hamming_graph_lambda2_power_iter,
)
from .block import SystroformerBlock
from .model import MiniSystroformer
from .utils import (
    LearnedAddressNet,
    lsh_subsample,
)

__all__ = [
    "address_from_activation",
    "derivative_catcher",
    "hamming_graph_lambda2",
    "hamming_graph_lambda2_power_iter",
    "SystroformerBlock",
    "MiniSystroformer",
    "LearnedAddressNet",
    "lsh_subsample",
]

__version__ = "0.1.0"
