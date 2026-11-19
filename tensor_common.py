"""GPU named-joint PD utilities for this checkpoint adapter."""
import json,re
from pathlib import Path
import torch
from torch_graph import TensorGraph
from motion_challenge.simulation.tensor_math import normalize,conjugate,rotate,multiply,matrix


def vector(value,names,device,default=0.):
    if isinstance(value,dict):
        result=[]
        for name in names:
            values=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
            if len(values)>1:raise ValueError('Ambiguous named joint value')
            result.append(values[0] if values else default)
        value=result
    value=torch.as_tensor(value,dtype=torch.float64,device=device)
    return value.expand(len(names)) if value.ndim==0 else value


def roll_pitch(q):
    m=matrix(q)
    return torch.stack((torch.atan2(m[...,2,1],m[...,2,2]),torch.asin((-m[...,2,0]).clamp(-1,1))),dim=-1)


class TensorPD:
    action_output='actions'
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        self.device=torch.device(device);self.slots=slots;self.runtime=runtime
        self.cfg=json.loads((Path(root)/'policy.json').read_text());self.names=self.cfg['policy_joint_names']
        self.order=[joint_names.index(n) for n in self.names];self.inverse=[self.names.index(n) for n in joint_names]
        self.default=vector(self.cfg['default_joint_pos'],self.names,device)
        self.scale=vector(self.cfg['action_scale'],self.names,device,1.)
        self.kp=vector(self.cfg['joint_kp'],self.names,device)[self.inverse];self.kd=vector(self.cfg['joint_kd'],self.names,device)[self.inverse]
        self.clip=self.cfg.get('action_clip');self.last=torch.zeros((slots,len(self.names)),device=device)
        self.fresh=torch.ones(slots,dtype=torch.bool,device=device)
        self.model=TensorGraph(converted_weights,device,runtime['weights_sha256']).eval()

    def reset(self,slots):
        self.last[slots]=0;self.fresh[slots]=True

    def proprio(self,state):
        q=state.joint_position_rad[:,self.order].double();qd=state.joint_velocity_rad_s[:,self.order].double()
        rotation=normalize(state.root_orientation_xyzw.double());inv=conjugate(rotation)
        omega=rotate(inv,state.root_angular_velocity_world_rad_s.double())
        gravity=rotate(inv,torch.tensor([0.,0.,-1.],dtype=torch.float64,device=self.device))
        return q,qd,omega,gravity,rotation

    @torch.inference_mode()
    def act(self,state,reference):
        raw=self.model(**self.observations(state,reference))[self.action_output]
        self.last.copy_(raw)
        action=raw.clamp(-self.clip,self.clip) if self.clip is not None else raw
        target=(self.default+self.scale*action)[:,self.inverse];zero=torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1).float()
