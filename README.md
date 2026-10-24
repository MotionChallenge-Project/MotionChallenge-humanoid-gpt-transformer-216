# Humanoid-GPT Transformer 216

[Author project](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) · [Checkpoint](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv216_trans.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-humanoid-gpt-transformer-216.git
cd MotionChallenge-humanoid-gpt-transformer-216
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo 3.17 CPU batch; `--jobs` sets parallelism. Missing data and weights are
downloaded and hash-checked; results and traces stay outside Git.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/85aae8656868779209b3f46696bc42883830ab3b/docs/OBSERVATIONS.md).

Author infer_onnx_real hardware route: fixed synthetic root [0,0,0.78], author reference FK, EMA and 16-frame history. No measured robot position or position integration.
