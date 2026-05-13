# Systrophe derived tools

This folder holds tools built **on top of** the Systrophe core
framework. Each tool lives in its own subdirectory and imports from
`systrophe` to reuse the address-space novelty catcher (`lambda_2`
algebraic-connectivity on Hamming-graph) or any other framework
primitive.

## Layout

```
tools/
├── README.md            <- you are here
├── systroformer/        <- λ₂-modulated FFN transformer (single-cylinder)
│   ├── README.md
│   ├── systroformer/    <- python package
│   │   ├── catcher.py   <- thin torch wrapper around systrophe.novelty_catcher
│   │   ├── block.py     <- SystroformerBlock
│   │   ├── model.py     <- MiniSystroformer
│   │   └── utils.py     <- LSH subsample + LearnedAddressNet
│   ├── experiments/
│   └── tests/
└── cyliformer/          <- Resonant Cylinder Transformer (N-cylinder, beam-form)
    ├── README.md
    ├── cyliformer/      <- python package
    │   ├── catcher.py   <- LearnedAddressCatcher (differentiable λ₂)
    │   ├── block.py     <- CylinderBlock (N phase-shifted views + soft prune)
    │   ├── model.py     <- Cyliformer LM + param breakdown
    │   ├── loss.py      <- cyliformer_loss + TorsionalResonanceLoss
    │   └── kv_cache.py  <- SelectiveKVCache
    ├── experiments/
    └── tests/
```

## Adding a new derived tool

1. Create `tools/<name>/` with the same layout as `systroformer/`.
2. Import from `systrophe.*` — do not re-implement catcher primitives.
3. Document what Systrophe concept the tool reuses and why.
4. Tests live in `tools/<name>/tests/`; the top-level
   `tests/` directory is reserved for the Systrophe framework.

## Why split?

The Systrophe package is research-grade and small (`pip install
systrophe` pulls only numpy/scipy). Derived tools may have heavier
dependencies (torch, transformers, datasets, ...) that we don't want
to force on framework-only users. Splitting keeps the core minimal and
lets each tool declare its own `requirements.txt`.
