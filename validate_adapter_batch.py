"""Validate full-capacity adapter state, forward and partial slot reset."""
import argparse
import json
from pathlib import Path
import time

import torch
from motion_challenge.sdk import DeployableStateBatch, MotionReferenceBatch
from motion_challenge.validation import sha256_file


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch', type=int, default=4096)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runtime = json.loads((root / 'runtime.mjwarp.json').read_text())
    if (root / 'policy.json').exists():
        cfg = json.loads((root / 'policy.json').read_text())
        policy_names = cfg['policy_joint_names']
    else:
        import yaml
        cfg = yaml.safe_load((root / 'policy.yaml').read_text())
        policy_names = cfg['joint_names']
    import tensor_adapter
    adapter_type = getattr(tensor_adapter, tensor_adapter.ADAPTER_CLASS)
    names = tuple(reversed(policy_names))
    n, device = args.batch, args.device
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.empty(0,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    rows = torch.tensor(sorted(set([0, 1, 2, n//4, n//2, 3*n//4, n-1])), device=device)
    full = adapter_type(root, runtime, names, n, device, args.converted)
    sampled = adapter_type(root, runtime, names, len(rows), device, args.converted)
    full.reset(torch.arange(n, device=device))
    sampled.reset(torch.arange(len(rows), device=device))
    generator = torch.Generator(device=device).manual_seed(12487)
    offsets = torch.tensor(runtime['reference_offsets_s'], device=device, dtype=torch.float64)
    k = len(offsets)
    max_error = 0.
    started = time.perf_counter()
    for step in range(4):
        def rand(*shape):
            return torch.randn(shape, generator=generator, device=device, dtype=torch.float64)
        q = .1 * rand(n,29)
        qd = .1 * rand(n,29)
        quat = .02 * rand(n,4)
        quat[:,3] += 1.
        quat /= quat.norm(dim=-1,keepdim=True)
        omega = .1 * rand(n,3)
        times = torch.full((n,), step*.02, device=device, dtype=torch.float64)
        state = DeployableStateBatch(names,times,q,qd,quat,omega)
        requested = times[:,None] + offsets[None]
        source = requested.clamp(min=0.)
        rj = .1 * rand(n,k,29)
        rp = .01 * rand(n,k,3)
        rp[...,2] += .78
        rq = .02 * rand(n,k,4)
        rq[...,3] += 1.
        rq /= rq.norm(dim=-1,keepdim=True)
        reference_values = (requested,source,rj,rp,rq,.1*rand(n,k,29),.1*rand(n,k,3),.1*rand(n,k,3))
        reference = MotionReferenceBatch(*reference_values)
        if step == 2:
            selection = torch.tensor([0,len(rows)//2,len(rows)-1],device=device)
            full.reset(rows[selection])
            sampled.reset(selection)
        actual = full.act(state,reference)
        subset_state = DeployableStateBatch(names,*[x[rows] for x in (times,q,qd,quat,omega)])
        subset_reference = MotionReferenceBatch(*[x[rows] for x in reference_values])
        expected = sampled.act(subset_state,subset_reference)
        assert actual.shape == (n,5,29)
        assert bool(torch.isfinite(actual).all())
        error = float((actual[rows]-expected).abs().max())
        max_error = max(max_error,error)
        torch.testing.assert_close(actual[rows],expected,atol=5e-5,rtol=2e-5)
    torch.cuda.synchronize(device)
    report = {
        'status':'passed','batch':n,'steps':4,'partial_reset_step':2,
        'sampled_rows':rows.tolist(),'max_pd_command_abs_error':max_error,
        'elapsed_s':time.perf_counter()-started,
        'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'graph_sha256':sha256_file(args.converted/'graph.json'),
        'adapter_source_sha256':{p.name:sha256_file(p) for p in root.glob('tensor*.py')},
        'reference':'4096-slot adapter versus independently reset 7-slot adapter; full forward without chunks',
    }
    destination = root/'validation/full-batch-adapter.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
