"""Render the production HIM controller and measure simulated foot contact; no DDS."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.him_qualification import sha256
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_mujoco.validate_go2_him import simulate

ROOT=Path(LEGGED_GYM_ROOT_DIR)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--title',default='HIM：当前权重，软件运动限制关闭；仿真录像无音轨')
    args=parser.parse_args();torch.set_num_threads(1)
    ffmpeg='/usr/bin/ffmpeg' if Path('/usr/bin/ffmpeg').exists() else shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg is required to record the comparison video')
    cfg=Go2RealConfig.load(args.config,str(ROOT))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=subprocess.Popen([ffmpeg,'-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','960x540','-framerate','25','-i','-','-an','-c:v','libx264','-preset','fast',
        '-crf','21','-pix_fmt','yuv420p',str(args.output)],stdin=subprocess.PIPE)
    renderer=None;camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.distance=1.25;camera.azimuth=135.;camera.elevation=-18.
    fontpath=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(fontpath),22) if fontpath.exists() else ImageFont.load_default()
    foot_ids=None;last_frame=-1;frames=0;telemetry=[]
    previous_contact=np.ones(4,dtype=bool);previous_vz=np.zeros(4);flight_start=[None]*4;touchdowns=[]

    def observe(model,data,t,command):
        nonlocal renderer,foot_ids,last_frame,frames,previous_contact,previous_vz
        if foot_ids is None:foot_ids=np.array([model.geom(name).id for name in ('FR','FL','RR','RL')])
        if 8.<=t<15.:
            normal=np.zeros(4);velocity=np.zeros(4)
            for i,gid in enumerate(foot_ids):
                vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,int(gid),vel,0)
                velocity[i]=vel[5]
            for i in range(data.ncon):
                contact=data.contact[i]
                for foot_index,gid in enumerate(foot_ids):
                    other=contact.geom2 if contact.geom1==gid else contact.geom1 if contact.geom2==gid else -1
                    if other>=0 and model.geom_bodyid[other]==0:
                        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                        normal[foot_index]+=abs(force[0])
            contact=normal>5.
            for i in range(4):
                if not contact[i] and flight_start[i] is None:flight_start[i]=t
                if contact[i] and not previous_contact[i]:
                    if flight_start[i] is not None and t-flight_start[i]>=.04:
                        touchdowns.append({'time':t,'leg':('FR','FL','RR','RL')[i],
                                           'incoming_vertical_speed_mps':max(0.,-float(previous_vz[i]))})
                    flight_start[i]=None
            previous_contact=contact;previous_vz=velocity.copy()
            telemetry.append([t,*normal,*velocity])
        if not 7.<=t<18.:return
        frame_index=int(round((t-7.)*500))//20
        if frame_index==last_frame:return
        last_frame=frame_index
        if renderer is None:
            model.vis.global_.offwidth=960;model.vis.global_.offheight=540
            renderer=mujoco.Renderer(model,height=540,width=960)
        camera.lookat[:]=data.qpos[:3];camera.lookat[2]=.18
        renderer.update_scene(data,camera=camera)
        frame=Image.fromarray(renderer.render())
        draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,68),fill=(20,24,30))
        title=args.title if fontpath.exists() else 'HIM: software motion limits disabled; silent simulation'
        draw.text((12,6),title,font=font,fill='white')
        draw.text((12,36),f't={t:.2f}s   vx={command[0]:.2f}  vy={command[1]:.2f}  yaw={command[2]:.2f}',font=font,fill='white')
        writer.stdin.write(np.asarray(frame).tobytes());frames+=1
    try:
        flow,_=simulate(cfg,'w',[.3,0,0],protected=True,observer=observe)
    finally:
        if renderer is not None:renderer.close()
        writer.stdin.close();status=writer.wait()
    if status or not frames:raise RuntimeError(f'Video encoder failed: status={status}, frames={frames}')
    values=np.asarray(telemetry)
    metrics={'simulation_only':True,'dds_initialized':False,'policy_sha256':sha256(cfg.policy_path),
        'config_sha256':cfg.source_sha256,'video_frames':frames,'fps':25,'video_window_seconds':[7,18],
        'motion_window_seconds':[8,15],'software_motion_limits_enabled':cfg.software_motion_limits_enabled,
        'leg_order':['FR','FL','RR','RL'],'peak_simulated_foot_normal_force_N':values[:,1:5].max(axis=0).tolist(),
        'foot_vertical_speed_min_mps':values[:,5:9].min(axis=0).tolist(),'touchdown_events':touchdowns,
        'flow':flow,'scope':'Force and foot speed are simulator quantities, not sound or measured hardware forces; touchdown events require at least 40ms of flight.'}
    args.output.with_suffix('.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),foot_contact_samples=values)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
