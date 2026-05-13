# Cyliformer

**Resonant Cylinder Transformer** — a transformer architecture derived
from Systrophe Phases 2a/2b/3a/3b.

Each block hosts **N "virtual cylinders"** that share a single FFN and
each rotate the input by a learnable phase angle (the cylinder's
phasor). A differentiable address-space λ₂ catcher measures how
coherently the cylinders resonate; a back-reaction proxy soft-prunes
(or hard-prunes at inference) cylinders whose coherence falls below a
target. The surviving cylinders are beam-summed into the block's
output. Provenance maps each design element to a Systrophe phase:

| Cyliformer component                | Systrophe phase                              |
|-------------------------------------|----------------------------------------------|
| Per-cylinder phasor                 | 3a (N-cylinder beam-forming)                 |
| Beam-sum of surviving cylinders     | 3a (`array_factor`, `beam_steer`)            |
| λ₂ catcher per cylinder             | 2a (Polyakov stress-energy diagnostic)       |
| Back-reaction soft prune            | 2b (Hadamard biparametrix bounded-V₁)        |
| Inference hard prune                | 3b (off-axis topology: prune disconnected)   |
| Cylinder "rotation" of channel pairs| 1+3 (van Stockum frame-dragging analog)      |

## Honest claims (parameter accounting)

- The shared FFN has the **same parameter count as a vanilla
  transformer's single FFN**. The savings claimed in
  `Cyliformer.txt` (50–87% reduction) are vs an
  **N-independent-FFNs-per-layer** baseline — the architecture whose
  *effective representational capacity* a Cyliformer with N cylinders
  *approximates*.
- Vs a standard transformer with a single FFN per layer, the
  parameter count is essentially identical. The Cyliformer's value
  proposition is **structural** (N rotated views, coherence gating,
  soft pruning), not raw parameter reduction.
- VRAM and quality claims (≈30% smaller, "matches a larger dense
  model") are **research conjectures** from the design notes; they
  have not been validated at scale in this repo. Tests here cover
  numerical stability, autograd correctness, and training convergence
  on a toy copy task — not benchmark-grade quality vs a vanilla model.

See `experiments/param_count_demo_results.json` for exact numbers.

## Layout

```
cyliformer/
├── README.md                        (you are here)
├── requirements.txt
├── cyliformer/
│   ├── __init__.py
│   ├── catcher.py                   # LearnedAddressCatcher (pytorch λ₂)
│   ├── block.py                     # CylinderBlock + CylinderBlockConfig
│   ├── model.py                     # Cyliformer + CyliformerConfig + param breakdown
│   ├── loss.py                      # cyliformer_loss + TorsionalResonanceLoss
│   └── kv_cache.py                  # SelectiveKVCache (selective per-cylinder KV)
├── experiments/
│   ├── train_copy.py                # copy-task training (~3s, loss 2.0 -> 0.003)
│   └── param_count_demo.py          # honest parameter-count comparison
└── tests/
    └── test_cyliformer.py           # 18 tests covering all public APIs
```

## Quick start

```bash
# from repo root
PYTHONPATH=src python tools/cyliformer/experiments/train_copy.py
```

Expected output (CPU, ~3 seconds):

```
Parameter breakdown:
  total: 27642
  shared_ffn: 16704         <- N cylinders all use these same weights
  attention: 8448
  catcher: 1584
  phasor: 10                <- 5 cylinders x 2 layers = 10 angles
  ...

epoch  0  loss=0.0066  lambda_per_layer=[...]  beam_per_layer=[...]
...
Final loss: 0.0033
```

## Block API

```python
from cyliformer import CylinderBlock, CylinderBlockConfig

block = CylinderBlock(CylinderBlockConfig(
    d_model=64, n_heads=4, n_cylinders=4,
    ffn_mult=4, lambda_target=0.20,
    backreaction_scale_init=0.15, prune_threshold=0.85,
    catcher_n_bits=24, catcher_radius=6,
    catcher_max_nodes=96, catcher_power_iter=6,
))

x = torch.randn(2, 16, 64)
y = block(x)                          # same shape as input
print(block.last_lambda2_per_cylinder)       # per-cylinder λ₂
print(block.last_backreaction_per_cylinder)  # per-cylinder back-reaction
print(block.beam_gain())                     # |Σ exp(i·δ_c)| / N
```

## Catcher

`LearnedAddressCatcher` is fully differentiable: it projects the
activations to a learned binary address, builds a Hamming-distance
graph, and returns λ₂ via power iteration on the graph Laplacian. The
projection weights see gradients on every forward pass, so the
catcher's notion of "address" co-adapts with the rest of the model.

## Loss functions

- `cyliformer_loss(logits, labels, lambdas, lambda_target, lambda_weight)`:
  cross-entropy + ReLU(target - λ₂)² penalty. Use as a drop-in
  replacement for `F.cross_entropy` during training.
- `TorsionalResonanceLoss`: full hybrid with derivative-smoothness and
  beam-alignment terms (see `Cyliformer.txt` for design intent).

## Provenance

Design from `Cyliformer.txt` (user's May-2026 design notes). The
implementation reuses Systrophe primitives where principled:
`systrophe.novelty_catcher` is the conceptual ancestor of
`LearnedAddressCatcher`, and the cylinder-phasor / beam-sum / soft-
prune structure mirrors the N-cylinder array work in
`src/systrophe/array.py` (Phase 3a).

## License

MIT, inherited from the Systrophe parent package.
