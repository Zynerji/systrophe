"""Pytest configuration for the brain-wallet auditor.

The Triton GPU-kernel correctness check (``kernels/test_triton_correctness.py``)
hard-imports ``triton`` (and a CUDA-capable ``torch``) at module import time.
On a host without those optional GPU dependencies, pytest would otherwise fail
*collection* of the whole tool directory with ``ModuleNotFoundError: No module
named 'triton'`` and exit with code 2 — even though the pure-Python CPU test
suite under ``tests/`` is completely independent of it.

Skip collecting the GPU kernel tests when Triton is unavailable so that
``pytest tools/brain-wallet-auditor`` runs the CPU suite cleanly. On a CUDA
host with Triton installed, the kernel tests are collected and run as normal.
"""

from importlib.util import find_spec

collect_ignore_glob = []

if find_spec("triton") is None:
    collect_ignore_glob.append("kernels/test_*.py")
