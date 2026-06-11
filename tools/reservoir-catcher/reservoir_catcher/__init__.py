"""reservoir-catcher — an honest physical-reservoir-computing test bench.

Tests two falsifiable questions about whether a self-organizing nonlinear
medium can integrate information / compute, and whether helical/Mobius
topology helps. See FINDINGS.md for the result (nonlinearity: yes;
designed topology: no).
"""
from .reservoir import make_W, run_reservoir, spectral_radius
from .tasks import narma10, parity_task, memory_capacity
from .readout import ridge_fit, ridge_pred, nrmse, address_lambda2
from .harness import evaluate, format_table, ARMS

__all__ = [
    "make_W", "run_reservoir", "spectral_radius",
    "narma10", "parity_task", "memory_capacity",
    "ridge_fit", "ridge_pred", "nrmse", "address_lambda2",
    "evaluate", "format_table", "ARMS",
]
