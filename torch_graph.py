"""Exact ONNX tensor weights with dynamic-batch PyTorch graph execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TensorGraph(torch.nn.Module):
    def __init__(self, directory, device='cpu', expected_source=None):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        directory=Path(directory)
        self.graph=json.loads((directory/'graph.json').read_text())
        if expected_source and self.graph['source_sha256']!=expected_source:raise ValueError('Source identity differs')
        if sha(directory/'weights.safetensors')!=self.graph['weights_sha256']:raise ValueError('Weights changed')
        self.keys={};self.integers={};self.shape_cache={}
        self.last_use={key:i for i,n in enumerate(self.graph['nodes']) for key in n['inputs']}
        self.outputs=set(self.graph['outputs'])
        for i,(name,value) in enumerate(load_file(str(directory/'weights.safetensors')).items()):
            if value.dtype in (torch.int32,torch.int64,torch.bool):self.integers[name]=value
            else:
                key=f'tensor_{i}';self.register_buffer(key,value.to(device));self.keys[name]=key

    def forward(self,**inputs):
        if set(inputs)!=set(self.graph['inputs']):raise ValueError('Input names differ')
        v={name:getattr(self,key) for name,key in self.keys.items()};v.update(self.integers);v.update(inputs)
        shape_key=tuple((k,tuple(x.shape)) for k,x in inputs.items());cache=self.shape_cache.setdefault(shape_key,{})
        for node_index,n in enumerate(self.graph['nodes']):
            out=n['outputs']
            if out[0] in cache:
                v.update(cache[out[0]]);continue
            x=[v[k] if k else None for k in n['inputs']];a=n['attributes'];op=n['op']
            def ints(t):return tuple(int(i) for i in t.reshape(-1).tolist())
            if op=='Identity':y=x[0]
            elif op=='Add':y=x[0]+x[1]
            elif op=='Sub':y=x[0]-x[1]
            elif op=='Mul':y=x[0]*x[1]
            elif op=='Div':y=x[0]/x[1]
            elif op=='Pow':y=x[0]**x[1]
            elif op=='MatMul':y=x[0]@x[1]
            elif op=='Gemm':
                left=x[0].T if a.get('transA',0) else x[0];right=x[1].T if a.get('transB',0) else x[1]
                y=a.get('alpha',1.)*(left@right)
                if len(x)>2:y=y+a.get('beta',1.)*x[2]
            elif op=='LayerNormalization':
                axis=a.get('axis',-1);y=F.layer_norm(x[0],x[0].shape[axis:],x[1],x[2] if len(x)>2 else None,a.get('epsilon',1e-5))
            elif op=='Concat':y=torch.cat(x,dim=a['axis'])
            elif op=='Sigmoid':y=x[0].sigmoid()
            elif op=='Tanh':y=x[0].tanh()
            elif op=='Mish':y=F.mish(x[0])
            elif op=='Relu':y=F.relu(x[0])
            elif op=='Elu':y=F.elu(x[0],alpha=a.get('alpha',1.))
            elif op=='Erf':y=x[0].erf()
            elif op=='Sqrt':y=x[0].sqrt()
            elif op=='Exp':y=x[0].exp()
            elif op=='Sin':y=x[0].sin()
            elif op=='Cos':y=x[0].cos()
            elif op=='Neg':y=-x[0]
            elif op=='Softmax':y=x[0].softmax(dim=a.get('axis',-1))
            elif op=='Shape':y=torch.tensor(tuple(x[0].shape)[a.get('start',0):a.get('end',x[0].ndim)],dtype=torch.int64)
            elif op=='Gather':
                axis=a.get('axis',0)%x[0].ndim;indices=x[1].long();indices=indices.remainder(x[0].shape[axis])
                y=torch.index_select(x[0],axis,indices.flatten().to(x[0].device)).reshape(tuple(x[0].shape[:axis])+tuple(indices.shape)+tuple(x[0].shape[axis+1:]))
            elif op=='Reshape':
                target=ints(x[1]);target=tuple(x[0].shape[i] if t==0 and not a.get('allowzero',0) else t for i,t in enumerate(target));y=x[0].reshape(target)
            elif op=='Transpose':y=x[0].permute(a.get('perm',list(reversed(range(x[0].ndim)))))
            elif op=='Flatten':axis=a.get('axis',1)%x[0].ndim;y=x[0].reshape(int(np.prod(x[0].shape[:axis])),int(np.prod(x[0].shape[axis:])))
            elif op=='Unsqueeze':
                y=x[0]
                axes=ints(x[1]) if len(x)>1 else a['axes'];rank=y.ndim+len(axes)
                for axis in sorted(k%rank for k in axes):y=y.unsqueeze(axis)
            elif op=='Squeeze':
                y=x[0];axes=ints(x[1]) if len(x)>1 else a.get('axes')
                if axes is None:y=y.squeeze()
                else:
                    for axis in sorted((k%y.ndim for k in axes),reverse=True):y=y.squeeze(axis)
            elif op=='Slice':
                starts,ends=ints(x[1]),ints(x[2]);axes=ints(x[3]) if len(x)>3 else range(len(starts));steps=ints(x[4]) if len(x)>4 else [1]*len(starts)
                y=x[0]
                for start,end,axis,step in zip(starts,ends,axes,steps):
                    start,end,step=slice(start,end,step).indices(y.shape[axis])
                    if step>0:
                        slices=[slice(None)]*y.ndim;slices[axis]=slice(start,end,step);y=y[tuple(slices)]
                    else:y=y.index_select(axis,torch.arange(start,end,step,device=y.device))
            elif op=='Split':
                sizes=ints(x[1]) if len(x)>1 else a.get('split');axis=a.get('axis',0)
                y=torch.split(x[0],sizes,dim=axis) if sizes else torch.chunk(x[0],len(out),dim=axis)
            elif op in ('ReduceMean','ReduceSum'):
                axes=ints(x[1]) if len(x)>1 else a.get('axes');axes=tuple(axes) if axes else tuple(range(x[0].ndim));keep=bool(a.get('keepdims',1))
                y=x[0].mean(axes,keepdim=keep) if op=='ReduceMean' else x[0].sum(axes,keepdim=keep)
            elif op=='Clip':y=x[0].clamp(min=x[1] if len(x)>1 else a.get('min'),max=x[2] if len(x)>2 else a.get('max'))
            elif op=='Cast':y=x[0].to({1:torch.float32,6:torch.int32,7:torch.int64,9:torch.bool,11:torch.float64}[a['to']])
            elif op=='Einsum':y=torch.einsum(a['equation'],*x)
            elif op=='Where':y=torch.where(x[0].to(x[1].device),x[1],x[2].to(x[1].device))
            elif op=='Conv':
                rank=x[0].ndim-2;pads=a.get('pads',[0]*(2*rank));z=x[0]
                if pads[:rank]!=pads[rank:]:z=F.pad(z,tuple(t for i in reversed(range(rank)) for t in (pads[i],pads[i+rank])));padding=0
                else:padding=tuple(pads[:rank])
                function={1:F.conv1d,2:F.conv2d,3:F.conv3d}[rank]
                y=function(z,x[1],x[2] if len(x)>2 else None,tuple(a.get('strides',[1]*rank)),padding,tuple(a.get('dilations',[1]*rank)),a.get('group',1))
            elif op=='GlobalAveragePool':y=x[0].mean(tuple(range(2,x[0].ndim)),keepdim=True)
            else:raise ValueError('Unsupported operator '+op)
            values=list(y) if isinstance(y,(tuple,list)) else [y]
            mapped=dict(zip(out,values));v.update(mapped)
            for key in n['inputs']:
                if key not in self.outputs and self.last_use.get(key)==node_index:v.pop(key,None)
            if all(z.device.type=='cpu' and z.dtype in (torch.int32,torch.int64,torch.bool) for z in values):cache[out[0]]=mapped
        return {k:v[k] for k in self.graph['outputs']}


def convert(source,directory,expected_hash,outputs):
    import onnx
    from onnx import numpy_helper,helper
    source,directory=Path(source),Path(directory)
    if sha(source)!=expected_hash:raise ValueError('Wrong original checkpoint')
    m=onnx.load(source);producers={o:n for n in m.graph.node for o in n.output};required=set()
    def visit(name):
        if name in required:return
        required.add(name)
        if name in producers:
            for arg in producers[name].input:visit(arg)
    for name in outputs:visit(name)
    weights={t.name:torch.from_numpy(numpy_helper.to_array(t).copy()) for t in m.graph.initializer if t.name in required}
    nodes=[]
    for n in m.graph.node:
        if not any(k in required for k in n.output):continue
        a={a.name:helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type=='Constant':
            weights[n.output[0]]=torch.from_numpy(numpy_helper.to_array(a['value']).copy());continue
        a={k:v.decode() if isinstance(v,bytes) else v for k,v in a.items()}
        nodes.append({'op':n.op_type,'inputs':list(n.input),'outputs':list(n.output),'attributes':a})
    directory.mkdir(parents=True,exist_ok=True);save_file(weights,str(directory/'weights.safetensors'))
    g={'source_sha256':expected_hash,'weights_sha256':sha(directory/'weights.safetensors'),'outputs':outputs,'nodes':nodes,'inputs':{v.name:[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim] for v in m.graph.input if v.name in required},'converter_sha256':sha(__file__)}
    (directory/'graph.json').write_text(json.dumps(g,indent=2)+'\n');return g


def validate(source,directory,device='cuda:1',batches=(1,7,40),atol=5e-5):
    from onnx.reference import ReferenceEvaluator
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ref=ReferenceEvaluator(str(source));g=json.loads((Path(directory)/'graph.json').read_text());rng=np.random.default_rng(279)
    model=TensorGraph(directory,device);rows=[];maximum=0.
    for batch in batches:
        feed={k:rng.normal(0,.5,size=(batch,*shape[1:])).astype(np.float32) for k,shape in g['inputs'].items()}
        expected={k:[] for k in g['outputs']}
        for i in range(batch):
            for k,v in zip(g['outputs'],ref.run(g['outputs'],{k:v[i:i+1] for k,v in feed.items()})):expected[k].append(v)
        expected={k:np.concatenate(v) for k,v in expected.items()}
        with torch.inference_mode():actual=model(**{k:torch.tensor(v,device=device) for k,v in feed.items()})
        errors={}
        for k,v in actual.items():
            v=v.cpu().numpy();np.testing.assert_allclose(v,expected[k],rtol=2e-5,atol=atol)
            errors[k]=float(np.max(abs(v-expected[k])));maximum=max(maximum,errors[k])
        rows.append({'batch':batch,'max_abs_error':errors})
    report={'status':'passed','source_sha256':sha(source),'weights_sha256':g['weights_sha256'],'graph_sha256':sha(Path(directory)/'graph.json'),'device':device,'batches':rows,'max_abs_error':maximum,'tolerance_abs':atol,'tolerance_rel':2e-5,'reference':'Original ONNX ReferenceEvaluator per sample'}
    (Path(directory)/'parity.json').write_text(json.dumps(report,indent=2)+'\n');return report
