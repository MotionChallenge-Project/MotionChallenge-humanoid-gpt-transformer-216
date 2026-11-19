# HumanoidGPT Transformer 216

GPU batch evaluation of the released HumanoidGPT Transformer 216 checkpoint on LAFAN and BONES-SEED.

## Run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-humanoid-gpt-transformer-216.git
cd MotionChallenge-humanoid-gpt-transformer-216
uv sync --locked
uv run evaluate.py --dataset lafan --data /path/to/lafan --device cuda:0 --output /path/to/results/humanoid-gpt-transformer-216-lafan
uv run evaluate.py --dataset seed --data /path/to/g1.tar.gz --device cuda:0 --output /path/to/results/humanoid-gpt-transformer-216-seed
```

Download original [LAFAN](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) or [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) data and accept the upstream terms. The LAFAN directory contains `g1/*.csv`. The SEED archive is selected by Core's single public inclusion list. Motion data stays local.

The command downloads and verifies the original model weights, converts them to safetensors plus a PyTorch execution graph, evaluates the complete selected dataset, and independently verifies the recorded trajectories. Use a fresh output directory for each run.

```bash
uv run evaluate.py --dataset lafan --data /path/to/lafan --weights /path/to/original-checkpoint --offline --device cuda:0 --output /path/to/results/humanoid-gpt-transformer-216-lafan
```

LAFAN uses 40 simulation slots and SEED uses 4096. Each control step runs one full-batch network forward. Finished slots are refilled in longest-motion-first order. `--slots` selects a different capacity. `--cache` sets the local preparation and model conversion cache.

BONES-SEED evaluation uses a feasibility-filtered subset of the original G1 motion data. Download the original archive; Core applies one public inclusion list of 129,561 motions shared by every model.

## Evaluation contract

The control period is 20 ms with four 5 ms physics substeps. Sliding friction is 1.0 and the Newton solver uses at most 50 iterations. Core precomputes scoring reference FK before rollout and reads actual body poses from MJWarp.

Policies receive joint positions and velocities, root orientation, angular velocity, and root/joint reference motion. Actual global position and actual linear velocity are unavailable. Model-specific reference features and policy history are maintained by this repository.

The original checkpoint SHA256 is `82313c76869ca796395abf80ee4288cced424d25087e74e32f9ec739ff59c853`. `runtime.mjwarp.json` specifies its source, input configuration, observation contract and conversion entrypoint.

## Validation

`validation/mjwarp-network-parity.json` records original ONNX versus PyTorch batch parity. `validation/full-batch-parity.json` records a 4096-sample forward and original ONNX row checks. `validation/tensor-observation-parity.json` records original observation/history/reset and PD-target parity.

Each evaluation writes `result.json`, trace files, input hashes and `verification.json`. Dataset scores require full motion coverage and independent trace verification.

`validation/full-batch-adapter.json` verifies 4096 active adapter states, a full forward and partial resets. `validation/onnxruntime-parity.json` checks the original ONNX in an independent ONNX Runtime environment.

<!-- current-evaluation-evidence -->

## Current evaluation evidence

[LAFAN1 full evaluation](validation/lafan-evaluation.json) · [BONES-SEED full evaluation](validation/seed-evaluation.json)
