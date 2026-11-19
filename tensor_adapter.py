"""HumanoidGPT's author deployment observations on GPU."""
from tensor_common import TensorPD,torch
from tensor_reference import TensorReferenceFeatures,quat_heading_wxyz


class TensorHumanoidGPT(TensorPD):
    action_output='continuous_actions'
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        super().__init__(root,runtime,joint_names,slots,device,converted_weights)
        self.reference=TensorReferenceFeatures(root,runtime,self.names,self.order,slots,device)
        self.history_length=runtime.get('history_len')
        self.history=torch.zeros((slots,self.history_length,136),device=device) if self.history_length else None

    def reset(self,slots):
        super().reset(slots);self.reference.reset(slots)

    def observations(self,state,reference):
        q,qd,omega,gravity,_=self.proprio(state)
        current,future=self.reference.update(reference)
        actual_heading=quat_heading_wxyz(state.root_orientation_xyzw[:,[3,0,1,2]].double())
        p=current[:,:3].double();c=actual_heading.cos();s=actual_heading.sin()
        position_error=torch.stack((c*p[:,0]+s*p[:,1],-s*p[:,0]+c*p[:,1]),dim=-1)
        heading_error=quat_heading_wxyz(current[:,3:7]).double()-actual_heading
        heading_error=torch.atan2(heading_error.sin(),heading_error.cos())
        command=torch.cat(((future[:,7:36]-self.default.float()).double(),future[:,2:3],future[:,36:45],heading_error.cos()[:,None],heading_error.sin()[:,None],position_error),dim=-1)
        obs=torch.cat((omega,gravity,q-self.default,qd,self.last,command),dim=-1).float()
        if self.history is not None:
            normal=torch.cat((self.history[:,1:],obs[:,None]),dim=1)
            self.history.copy_(torch.where(self.fresh[:,None,None],obs[:,None],normal));obs=self.history.clone()
        self.fresh.zero_()
        return {'obs':obs}


ADAPTER_CLASS='TensorHumanoidGPT'
NATIVE_CLASS='HumanoidGPT'
