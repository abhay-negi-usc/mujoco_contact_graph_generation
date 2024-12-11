import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
import os 
import pickle 
from scipy.spatial.transform import Rotation as R 

# go through each timestep of each trial of data 

dir_results = "/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/cross_peg_data_v3" 
dir_pkl = dir_results + "/pkl" 
pkl_files = sorted(glob.glob(os.path.join(dir_pkl, "*.pkl")), key=os.path.getmtime)

dir_save = dir_pkl.removesuffix("pkl") + "processed_data"
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

# list of all contact state history 
N_timesteps = 300  
N_trials = 10_000 
N_trials = len(pkl_files) if len(pkl_files) < N_trials else N_trials
pkl_files = pkl_files[:N_trials] 
pose_boundary_list = [] 

for i, pkl_file in enumerate(pkl_files): 

    # Read the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # unpack data 
    state_hist = data['state_hist'] 
    contact_pos = data['contact_pos'] 

    # check if there is contact within the hole area, if so, add the pose to the list 
    for j, contact_pos_j in enumerate(contact_pos): # iterate through each time step 
        if len(contact_pos_j) > 0: # if there is contact 
            for k, contact_pos_hole_frame in enumerate(contact_pos_j): # iterate through each contact at current time step 
                if contact_pos_hole_frame[2] < 0 and max(np.abs(contact_pos_hole_frame[:2])) < 0.012: # if contact is below the surface and within hole area 
                    peg_pose = state_hist[j, 1:8] 
                    pose_boundary_list.append(peg_pose) 
                    continue # don't need to check pose again 

    # print progress rate every 10% of total iterations 
    if (i+1) % np.floor(len(pkl_files)/10) == 0: 
        print(f"Completion Progress: {i+1}/{len(pkl_files)}")  

# convert list to dataframe 
pose_boundary_df = pd.DataFrame(pose_boundary_list, columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']) 

# convert quaternion to euler angles 
quaternions = pose_boundary_df[['qw', 'qx', 'qy', 'qz']].values 
euler_angles = R.from_quat(quaternions, scalar_first=True).as_euler("xyz", degrees=True) 
pose_boundary_df['a'] = euler_angles[:,2]
pose_boundary_df['b'] = euler_angles[:,1]
pose_boundary_df['c'] = euler_angles[:,0] 

# convert position from meters to millimeters 
pose_boundary_df[['x', 'y', 'z']] *= 1000 

# save the dataframe 
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)
pose_boundary_df.to_csv(os.path.join(dir_save, "pose_boundary_data_10k.csv"), index=False)  

print(f"Pose boundary data saved to {os.path.join(dir_save, 'pose_boundary_data_10k.csv')}.") 
print("Number of data points: ", len(pose_boundary_df)) 
