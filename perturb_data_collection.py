import mujoco 
import mediapy as media 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R 
import pickle 
import os 
import cv2 

geometry = "cross" # "extrusion" "cross" "plug_3_pin" 
dir_results = f"./results/{geometry}_data/perturb"  
xml_path = f"./env/{geometry}_env.xml" 
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
num_trials = 25 # 1k points ~ 27 minutes with videos, 30 seconds without videos 
flag_show_video = False
flag_save_video = True 

os.makedirs(dir_results + "/pkl", exist_ok=True)
if flag_save_video: 
    os.makedirs(dir_results + "/vid", exist_ok=True)

n_frames = 300   
height = 720 
width = 960
frames = []

# visualize contact frames and forces, make body transparent
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False 
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.02
model.vis.scale.contactheight = 0.02
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3
model.opt.gravity = (0,0,0)

for idx_trial in range(num_trials): 

    # define initial conditions 
    x0 = np.random.uniform(-3, +3) * (1e-3) 
    y0 = np.random.uniform(-3, +3) * (1e-3) 
    z0 = np.random.uniform(+25, +30) * (1e-3) 
    a0 = np.random.uniform(-5.0, +5.0) * (np.pi/180)  
    b0 = np.random.uniform(-5.0, +5.0) * (np.pi/180)  
    c0 = np.random.uniform(-5.0, +5.0) * (np.pi/180) 

    mujoco.mj_resetData(model, data)
    data.qpos = np.array([x0, y0, z0, a0, b0, c0]) 
    data.qvel = np.zeros(6) 
    mujoco.mj_forward(model, data)

    # initialize data structures 
    frames = []
    state_hist = np.zeros((n_frames,1+3+4+6))   
    contact_hist = [] 
    contact_num = [] 
    contact_geom1 = [] 
    contact_geom2 = [] 
    contact_dist = [] 
    contact_pos = [] 
    contact_frame = [] 
    ctrl_hist = np.zeros((n_frames,1+6))   
    sensor_hist = np.zeros((n_frames,13))

    # define controller parameters  
    z_step = 1e-6

    if flag_save_video: 
        # Initialize video writer
        video_path = dir_results + f"/vid/trial_{idx_trial}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    # Simulate and display video.
    with mujoco.Renderer(model, height, width) as renderer:
        for i in range(n_frames): 
            while data.time < i/(30.0*4): #1/4x real time
                mujoco.mj_step(model, data)
            if flag_show_video or flag_save_video: 
                renderer.update_scene(data, "track", options)
                frame = renderer.render()
                frames.append(frame)

                # Convert frame to BGR format for OpenCV
                if flag_save_video:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

            # save data 
            state_hist[i,:] = np.concatenate([np.array([data.time]), data.xpos[2], data.xquat[2], data.qvel]) 
            contact_num.append(len(data.contact.geom1)) 
            contact_geom1.append(np.array(data.contact.geom1)) 
            contact_geom2.append(np.array(data.contact.geom2)) 
            contact_dist.append(np.array(data.contact.dist))  
            contact_pos.append(np.array(data.contact.pos)) 
            contact_frame.append(np.array(data.contact.frame)) 
            ctrl_hist[i,:] = np.concatenate([np.array([data.time]), data.ctrl])  
            sensor_hist[i,:] = data.sensordata  

            # controller update 
            peg_x_axis = data.xmat[2].reshape(3,3)[:,0] 
            peg_y_axis = data.xmat[2].reshape(3,3)[:,1] 
            peg_z_axis = data.xmat[2].reshape(3,3)[:,2]  
            angle_step = 1.0 * np.pi/180 
            ii = i * angle_step 
            theta = 3 * ii 
            phi = 5 * ii 
            psi = 7 * ii 
            tau = ii 
            r = 1.0e-9 
            delta_x = r * np.cos(tau) * peg_x_axis 
            delta_y = r * np.sin(tau) * peg_y_axis 
            delta_z = -i * z_step * peg_z_axis 
            delta_pos = delta_x + delta_y + delta_z 
            delta_a = 5.0 * np.sin(theta)
            delta_b = 5.0 * np.sin(phi)
            delta_c = 5.0 * np.sin(psi) 
            delta_angle = np.array([delta_a, delta_b, delta_c]) * np.pi/180 
            delta_pose_tool = np.concatenate([delta_pos, delta_angle]) 
            data.ctrl = data.qpos + delta_pose_tool 
                
        if flag_show_video: 
            media.show_video(frames, fps=30)

        if flag_save_video:
            # Release video writer
            video_writer.release() 

        data_dict = {
            'state_hist': state_hist, 
            'contact_num': contact_num,   
            'contact_geom1': contact_geom1,
            'contact_geom2': contact_geom2,
            'contact_dist': contact_dist,
            'contact_pos': contact_pos,
            'contact_frame': contact_frame,
            'ctrl_hist': ctrl_hist,
            'sensor_hist': sensor_hist,
        }

        # save data as pkl file 
        with open(dir_results + f"/pkl/trial_{idx_trial}.pkl", 'wb') as f: 
            pickle.dump(data_dict, f) 
        
        print(f"Trial {idx_trial} complete.") 