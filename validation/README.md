# Author inference comparison

`validate_inference.py` compares pinned author code with this adapter on 160 matched inputs, four resets and 32 endpoint cases. Source files and checkpoint bytes are checked against `native-sources.json`.

The released Transformer checkpoint requires 16 × 136 inputs. History uses the author's repeat-first-frame initialization and chronological append order. Reference EMA, FK, observation construction, ONNX output, raw-action history and motor targets are compared at tolerance 2e-5. Reports: `native-parity.json` (macOS) and `native-parity-linux.json` (Linux). These comparisons are separate from full motion evaluation.

```bash
uv run validate_inference.py --source /path/to/pinned/author/source --weights /path/to/pns_wo_priv216_trans.onnx --output /tmp/hgpt216-native-parity.json
```

Benchmark physics, initial conditions, failures and metrics remain owned by MotionChallenge-Core. See `provenance.json` for input-only reference preprocessing.
