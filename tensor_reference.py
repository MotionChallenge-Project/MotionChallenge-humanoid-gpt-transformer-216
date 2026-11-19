"""Author EMA and private reference kinematics for batched HumanoidGPT."""
from pathlib import Path
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from motion_challenge.validation import sha256_file
from motion_challenge.simulation.tensor_math import normalize,rotate,quaternion_velocity


def quat_heading_wxyz(q):
    w,x,y,z=q.unbind(-1)
    angle=torch.atan2(2*(x*y+w*z),1-2*(y*y+z*z))
    return torch.atan2(angle.sin(),angle.cos())


class TensorReferenceFeatures:
    def __init__(self,root,runtime,names,order,slots,device):
        self.device=torch.device(device);self.slots=slots;self.order=order;self.dt=runtime['control_period_s']
        path=Path(root)/runtime['kinematics']
        if sha256_file(path)!=runtime['kinematics_sha256']:raise ValueError('Reference kinematics asset changed')
        self.cpu=mujoco.MjModel.from_xml_path(str(path))
        self.qids=[int(self.cpu.joint(n).qposadr[0]) for n in names];self.vids=[int(self.cpu.joint(n).dofadr[0]) for n in names]
        self.pelvis=self.cpu.body('pelvis').id
        with wp.ScopedDevice(str(device)),self.stream():
            self.model=mjw.put_model(self.cpu);self.data=mjw.make_data(self.cpu,nworld=slots,nconmax=1,njmax=1)
            self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel)
            self.qpos[:]=torch.as_tensor(self.cpu.qpos0,device=device,dtype=torch.float32)
        self.previous=torch.zeros((slots,36),device=device)
        self.has_previous=torch.zeros(slots,device=device,dtype=torch.bool)
        self.keys=torch.zeros((slots,2),device=device,dtype=torch.int64)
        self.valid=torch.zeros((slots,2),device=device,dtype=torch.bool)
        self.values=torch.zeros((slots,2,45),device=device)

    def stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def reset(self,slots):
        self.has_previous[slots]=False;self.valid[slots]=False

    def compute(self,raw):
        previous=self.previous
        quat=torch.where(((previous[:,3:7]*raw[:,3:7]).sum(-1)<0)[:,None],-raw[:,3:7],raw[:,3:7])
        smoothed=.8*previous+.2*raw
        blended=.8*previous[:,3:7]+.2*quat
        smoothed[:,3:7]=blended/torch.linalg.vector_norm(blended,dim=-1,keepdim=True).clamp_min(1e-8)
        smoothed=torch.where(self.has_previous[:,None],smoothed,raw)
        normalized=smoothed.double();normalized[:,3:7]=normalize(normalized[:,3:7])
        old=normalize(previous[:,[4,5,6,3]].double());new=normalized[:,[4,5,6,3]]
        velocity=torch.zeros((self.slots,35),device=self.device,dtype=torch.float64)
        velocity[:,:3]=(normalized[:,:3]-previous[:,:3])/self.dt
        velocity[:,3:6]=rotate(old,quaternion_velocity(old,new,torch.full((self.slots,),self.dt,device=self.device,dtype=torch.float64)))
        velocity[:,6:]=(normalized[:,7:]-previous[:,7:])/self.dt
        velocity=torch.where(self.has_previous[:,None],velocity,0.)
        self.qpos[:,:7]=normalized[:,:7];self.qpos[:,self.qids]=normalized[:,7:].float()
        self.qvel[:,:6]=velocity[:,:6];self.qvel[:,self.vids]=velocity[:,6:].float()
        with wp.ScopedDevice(str(self.device)),self.stream():
            mjw.kinematics(self.model,self.data);mjw.com_pos(self.model,self.data);mjw.com_vel(self.model,self.data)
        cv=wp.to_torch(self.data.cvel)[:,self.pelvis]
        gravity=-wp.to_torch(self.data.xmat)[:,self.pelvis,2,:]
        # Author heading rotation is computed in float64, rounded to float32.
        yaw=quat_heading_wxyz(normalized[:,3:7]);c=yaw.cos().float();s=yaw.sin().float()
        def heading_inverse(v):return torch.stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]),dim=-1)
        feature=torch.cat((normalized.float(),gravity,heading_inverse(cv[:,:3]),heading_inverse(cv[:,3:])),dim=-1)
        return smoothed,feature

    def update(self,reference):
        if reference.source_time_s.shape[1]!=2:raise ValueError('HumanoidGPT expects current and future references')
        keys=torch.round(reference.source_time_s*1e9).long()
        old_keys,old_valid,old_values=self.keys,self.valid,self.values
        result=[]
        for frame in range(2):
            key=keys[:,frame]
            matches=(old_keys==key[:,None])&old_valid
            found=matches.any(-1);index=matches.long().argmax(-1)
            cached=old_values[torch.arange(self.slots,device=self.device),index]
            if frame:
                same=key==keys[:,0];cached=torch.where(same[:,None],result[0],cached);found|=same
            raw=torch.cat((reference.root_position_world_m[:,frame],reference.root_orientation_xyzw[:,frame,[3,0,1,2]],reference.joint_position_rad[:,frame,self.order]),dim=-1).float()
            smoothed,feature=self.compute(raw)
            self.previous.copy_(torch.where(found[:,None],self.previous,smoothed));self.has_previous|=~found
            result.append(torch.where(found[:,None],cached,feature))
        self.keys=keys;self.valid=torch.ones_like(old_valid);self.values=torch.stack(result,dim=1)
        return result
