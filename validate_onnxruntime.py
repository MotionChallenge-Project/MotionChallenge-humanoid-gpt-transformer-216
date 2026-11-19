"""Independent ONNX Runtime check using an isolated CPU Python environment."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from torch_graph import TensorGraph, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--converted',type=Path,required=True)
    parser.add_argument('--ort-python',required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    graph=json.loads((args.converted/'graph.json').read_text())
    assert sha(args.source)==graph['source_sha256']
    rng=np.random.default_rng(9927)
    feed={k:rng.normal(0,.5,(7,*shape[1:])).astype(np.float32) for k,shape in graph['inputs'].items()}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=TensorGraph(args.converted,args.device,graph['source_sha256'])
    with torch.inference_mode():
        actual={k:v.cpu().numpy() for k,v in model(**{k:torch.tensor(v,device=args.device) for k,v in feed.items()}).items()}
    program='''import json,sys,numpy as np,onnxruntime as ort
source,feed_path,outputs_path,result_path=sys.argv[1:]
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
session=ort.InferenceSession(source,sess_options=options,providers=['CPUExecutionProvider'])
feed=dict(np.load(feed_path));names=json.loads(open(outputs_path).read());result={name:[] for name in names}
for row in range(7):
 for name,value in zip(names,session.run(names,{k:v[row:row+1] for k,v in feed.items()})):result[name].append(value)
np.savez(result_path,**{k:np.concatenate(v) for k,v in result.items()})
print(json.dumps({'onnxruntime':ort.__version__,'providers':session.get_providers()}))
'''
    with tempfile.TemporaryDirectory(prefix='motionchallenge-ort-') as temporary:
        p=Path(temporary);np.savez(p/'feed.npz',**feed);(p/'outputs.json').write_text(json.dumps(graph['outputs']))
        run=subprocess.run([args.ort_python,'-c',program,str(args.source),str(p/'feed.npz'),str(p/'outputs.json'),str(p/'result.npz')],check=True,capture_output=True,text=True)
        oracle=dict(np.load(p/'result.npz'));metadata=json.loads(run.stdout)
    errors={}
    for name,value in actual.items():
        np.testing.assert_allclose(value,oracle[name],atol=5e-5,rtol=2e-5)
        errors[name]=float(np.max(abs(value-oracle[name])))
    report={'status':'passed','batch':7,'original_onnx_calls':7,'device':args.device,'oracle':metadata,
            'max_abs_error':errors,'source_sha256':sha(args.source),'graph_sha256':sha(args.converted/'graph.json'),'weights_sha256':graph['weights_sha256']}
    (root/'validation/onnxruntime-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
