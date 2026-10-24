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
MuJoCo 3.17 CPU batch; `--jobs 8` sets parallelism. Missing data and weights are
downloaded and hash-checked; results and traces stay outside Git.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observation track

The restricted observation track is not yet admitted for this checkpoint. [Admission and special handling](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).
