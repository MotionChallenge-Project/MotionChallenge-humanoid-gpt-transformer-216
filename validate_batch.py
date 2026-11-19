"""Run one full 4096-sample forward and audit selected rows against original ONNX."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from torch_graph import TensorGraph,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--converted',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=4096)
    a=p.parse_args();root=Path(__file__).resolve().parent;g=json.loads((a.converted/'graph.json').read_text())
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(227);feed={k:rng.normal(0,.5,(a.batch,*s[1:])).astype(np.float32) for k,s in g['inputs'].items()}
    model=TensorGraph(a.converted,a.device,g['source_sha256'])
    inputs={k:torch.tensor(v,device=a.device) for k,v in feed.items()}
    torch.cuda.reset_peak_memory_stats(a.device);torch.cuda.synchronize(a.device);started=time.monotonic()
    with torch.inference_mode():result=model(**inputs)
    torch.cuda.synchronize(a.device);elapsed=time.monotonic()-started
    actual={k:v.cpu().numpy() for k,v in result.items()}
    if any(not np.isfinite(v).all() for v in actual.values()):raise ValueError('Nonfinite batched outputs')
    selected=np.unique(np.r_[0,a.batch-1,rng.choice(a.batch,min(32,a.batch),replace=False)])
    oracle=ReferenceEvaluator(str(a.source));maximum=0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for i in selected:
            expected=oracle.run(g['outputs'],{k:v[i:i+1] for k,v in feed.items()})
            for k,v in zip(g['outputs'],expected):
                np.testing.assert_allclose(actual[k][i:i+1],v,rtol=2e-5,atol=5e-5)
                maximum=max(maximum,float(np.max(abs(actual[k][i:i+1]-v))))
    report={'status':'passed','batch':a.batch,'network_forward_calls':1,'device':a.device,'sampled_original_onnx_rows':selected.tolist(),'max_abs_error':maximum,
            'full_forward_s':elapsed,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(a.device),'source_sha256':sha(a.source),'graph_sha256':sha(a.converted/'graph.json'),'weights_sha256':g['weights_sha256']}
    (root/'validation/full-batch-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
