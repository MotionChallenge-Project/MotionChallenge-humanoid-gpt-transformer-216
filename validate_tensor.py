"""Independent original-adapter observations and ONNX output parity."""
import argparse,json,sys,types
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from motion_challenge.validation import sha256_file
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket,MotionReferenceBatch,ReferenceFrame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--converted',type=Path,required=True);parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=json.loads((root/'runtime.mjwarp.json').read_text());cfg=json.loads((root/'policy.json').read_text())
    from tensor_adapter import ADAPTER_CLASS,NATIVE_CLASS
    import tensor_adapter
    tensor_type=getattr(tensor_adapter,ADAPTER_CLASS)
    if 'onnxruntime' not in sys.modules:sys.modules['onnxruntime']=types.ModuleType('onnxruntime')
    import adapter as original_module
    from motion_challenge.adapters.onnx import vector
    original_type=getattr(original_module,NATIVE_CLASS)
    # Use reversed named order to exercise both mappings.
    joint_names=tuple(reversed(cfg['policy_joint_names']));slots=7;device=args.device
    batched=tensor_type(root,runtime,joint_names,slots,device,args.converted)
    batched.reset(torch.arange(slots,device=device));original=[]
    for _ in range(slots):
        a=original_type.__new__(original_type);a.cfg=cfg;a.runtime=runtime;a.names=cfg['policy_joint_names'];a.default=vector(cfg['default_joint_pos'],a.names)
        a.scale=vector(cfg['action_scale'],a.names,1.)
        from reference import ReferenceFeatures
        a.reference=ReferenceFeatures(root,runtime,a.names)
        if runtime.get('history_len'):a.history_len=runtime['history_len'];a.history=None
        original.append(a)
    oracle=ReferenceEvaluator(str(args.weights));rng=np.random.default_rng(7913);max_observation=0.;max_action=0.;max_target=0.
    for step in range(24):
        q=rng.normal(0,.2,(slots,29));qd=rng.normal(0,.2,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);omega=rng.normal(0,.3,(slots,3))
        offsets=np.asarray(runtime['reference_offsets_s']);k=len(offsets);requested=step*.02+np.broadcast_to(offsets,(slots,k));times=requested.clip(0,.37)
        rj=rng.normal(0,.2,(slots,k,29));rp=rng.normal(0,.2,(slots,k,3));rp[...,2]+=.75;rq=rng.normal(size=(slots,k,4));rq/=np.linalg.norm(rq,axis=-1,keepdims=True)
        jv=rng.normal(0,.2,(slots,k,29));lv=rng.normal(0,.2,(slots,k,3));av=rng.normal(0,.2,(slots,k,3))
        def tensor(x):return torch.tensor(np.array(x),device=device)
        state=DeployableStateBatch(joint_names,tensor(np.full(slots,step*.02)),*[tensor(x) for x in (q,qd,quat,omega)])
        reference=MotionReferenceBatch(*[tensor(x) for x in (requested,times,rj,rp,rq,jv,lv,av)])
        if step==11:batched.reset(torch.tensor([1,4],device=device))
        got=batched.observations(state,reference);expected_actions=[]
        for row,a in enumerate(original):
            packet=DeployableStatePacket(step*.02,joint_names,tuple(q[row]),tuple(qd[row]),tuple(quat[row]),tuple(omega[row]))
            frames=tuple(ReferenceFrame(float(requested[row,i]),float(times[row,i]),tuple(rj[row,i]),tuple(rp[row,i]),tuple(rq[row,i]),tuple(jv[row,i]),tuple(lv[row,i]),tuple(av[row,i])) for i in range(k))
            if step==0 or (step==11 and row in (1,4)):a.reset(packet,frames,0)
            expected={key:np.asarray(value,np.float32) for key,value in a.observations(packet,frames).items()}
            for key,value in got.items():
                actual=value[row:row+1].cpu().numpy();err=float(np.max(abs(actual-expected[key])));max_observation=max(max_observation,err)
                tolerance=np.full_like(expected[key],2e-6);tolerance[...,126:132]=2e-5
                if not np.all(abs(actual-expected[key])<=tolerance+2e-6*abs(expected[key])):raise AssertionError(f'{key} observation error {err}')
            action=oracle.run([a.action_output],expected)[0].reshape(-1);expected_actions.append(action);a.last=action.astype(np.float32)
        with torch.inference_mode():actual=batched.model(**got)[batched.action_output]
        expected=np.stack(expected_actions);value=actual.cpu().numpy();max_action=max(max_action,float(np.max(abs(value-expected))))
        np.testing.assert_allclose(value,expected,atol=5e-5,rtol=2e-5)
        clip=cfg.get('action_clip');a_value=np.clip(value,-clip,clip) if clip is not None else value;e_value=np.clip(expected,-clip,clip) if clip is not None else expected
        target_error=float(np.max(abs((a_value-e_value)*original[0].scale)));max_target=max(max_target,target_error)
        if target_error>5e-5:raise AssertionError('Joint position target error exceeds 5e-5 rad')
        # Same raw action histories isolate the observation and inference tests.
        batched.last.copy_(tensor(expected))
    report={'checkpoint_sha256':sha256_file(args.weights),'adapter_source_sha256':{p.name:sha256_file(p) for p in root.glob('tensor*.py')},'graph_sha256':sha256_file(args.converted/'graph.json'),'status':'passed','steps':24,'batch':slots,'joint_order':'reversed canonical names','partial_reset_step':11,'partial_reset_slots':[1,4],
            'observation_max_abs_error':max_observation,'network_max_abs_error':max_action,'joint_target_max_abs_error_rad':max_target,
            'reference':'Original adapter.py observations and original ONNX per sample','history':'Matched raw actions isolate observation and inference parity','observation_tolerance':{'reference_cvel_absolute':2e-5,'other_absolute':2e-6,'relative':2e-6}}
    (root/'validation').mkdir(exist_ok=True);(root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
