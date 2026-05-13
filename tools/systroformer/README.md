# Systroformer

A transformer block whose FFN output is modulated by the algebraic
connectivity (λ₂) of the Hamming graph of its own address-hashed
attention activations.

Built on the Systrophe core framework
(`systrophe.novelty_catcher`). The Systrophe catcher detects
information-topological structure in any array of measurements;
applying it to transformer attention activations gives the model an
explicit global structural-novelty signal it can learn to amplify or
dampen.

## Layout

```
systroformer/
├── systroformer/
│   ├── catcher.py     # power-iter λ₂ + thin wrapper over systrophe.novelty_catcher
│   ├── block.py       # SystroformerBlock = attention + FFN + λ₂ modulation
│   ├── model.py       # MiniSystroformer (stacked blocks)
│   └── utils.py       # LSH subsample + LearnedAddressNet (straight-through)
├── experiments/
│   ├── train_copy.py          # end-to-end training on copy task
│   └── scalability_test.py    # catcher overhead at multiple (bs, seq_len)
├── tests/
│   └── test_systroformer.py   # 10 unit tests
└── requirements.txt
```

## Quick start

```bash
# from repo root
pip install torch numpy
PYTHONPATH=src python tools/systroformer/experiments/train_copy.py
```

## Block API

```python
from systroformer import SystroformerBlock

block = SystroformerBlock(
    d_model=64, n_heads=4,
    radius=5,           # Hamming radius for the address graph
    n_bits=32,          # bits per address
    lambda_scale_init=0.05,
    max_nodes=256,      # cap graph size via LSH-style subsample
    approximate_lambda2=True,  # use power-iteration instead of full eigvalsh
)
x = torch.randn(2, 16, 64)
y = block(x)                          # standard transformer-block shape
print(block.last_lambda2)             # most recent λ₂ value
print(block.last_derivative)          # most recent λ₂ derivative
```

## Provenance

This tool is the first concrete neural-net application of the
Systrophe address-space catcher to LLM-architecture research. The
design follows the prototype described in `SystropheLLMhelper.txt`
(user's notes, May 2026). The framework primitives —
`real_array_to_address`, `lambda_2_of_hamming_graph`,
`hamming_distance` — are re-exported from `systrophe.novelty_catcher`
to avoid duplication.

## Status

- 10 unit tests passing.
- End-to-end training on copy task (8-token seq, vocab 10, 2 layers)
  converges to copy-loss < 0.1 in ~5 epochs on CPU.
- Catcher overhead at (batch=4, seq_len=64): ~5x naive forward,
  expected to drop with LSH + learned address net.

## License

MIT, inherited from the Systrophe parent package.
