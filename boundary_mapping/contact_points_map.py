import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
import os 
import pickle 
from scipy.spatial.transform import Rotation as R 
import random 

# go through each timestep of each trial of data 

geometry = "square_B" # "extrusion" "cross" "plug_3_pin" "square_*"

dir_results = f"/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/{geometry}_data/perturb_v3" 
dir_pkl = dir_results + "/pkl" 
pkl_files = sorted(glob.glob(os.path.join(dir_pkl, "*.pkl")), key=os.path.getmtime)

dir_save = dir_pkl.removesuffix("pkl") + "processed_data"
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

# list of all contact state history 
N_timesteps = 500  
N_trials = 10_000 # FIXME: program gets killed if N_trials is 1_000_000 
N_trials = len(pkl_files) if len(pkl_files) < N_trials else N_trials
pkl_files = [pkl_files[i] for i in random.sample(range(len(pkl_files)), N_trials)]
contact_point_list = [] 

z_tol_top_surf = 1.0 # mm 

output_file = f"{geometry}_peg_contact_points_{N_trials}.csv"

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
                if geometry == "cross":
                    if contact_pos_hole_frame[2] < 0.0001 and max(contact_pos_hole_frame[:2]) < 0.012: # if contact is below the surface and within hole area 
                        contact_point_list.append(contact_pos_hole_frame) 
                        continue # don't need to check pose again 
                elif geometry == "plug_3_pin": 
                    x_contact = contact_pos_hole_frame[0] * 1E3
                    y_contact = contact_pos_hole_frame[1] * 1E3
                    z_contact = contact_pos_hole_frame[2] * 1E3 
                    if z_contact < z_tol_top_surf and (((x_contact-0)**2 + (y_contact - -10)**2 < 8) or ((x_contact - -6)**2 + (y_contact - 3.5)**2 < 10) or ((x_contact - 6)**2 + (y_contact - 3.5)**2 < 8)): # if contact is below the surface and within hole area 
                        contact_point_list.append(contact_pos_hole_frame) 
                        continue # don't need to check pose again 
                if geometry == "extrusion":
                    if contact_pos_hole_frame[2] < 0.0001 and np.max(np.abs(contact_pos_hole_frame[:2])) < 0.012: # if contact is below the surface and within hole area # NOTE: the 12mm limit may have been too tight 
                        contact_point_list.append(contact_pos_hole_frame) 
                        continue # don't need to check pose again 
                if "square" in geometry: 
                    if contact_pos_hole_frame[2] < 0.0001 and np.max(np.abs(contact_pos_hole_frame[:2])) < 0.0135: # if contact is below the surface and within hole area 
                        contact_point_list.append(contact_pos_hole_frame) 
                        continue # don't need to check pose again
                if "circle" in geometry:  
                    if contact_pos_hole_frame[2] < 0.0001 and np.linalg.norm(contact_pos_hole_frame[:2]) < 0.0135: # if contact is below the surface and within hole area 
                        contact_point_list.append(contact_pos_hole_frame) 
                        continue # don't need to check pose again
    
    # print progress rate every 10% of total iterations 
    if (i+1) % np.floor(len(pkl_files)/100) == 0: 
        print(f"Completion Progress: {i+1}/{len(pkl_files)}")  

# convert list to np array 
contact_points = np.array(contact_point_list) 
# convert to mm 
contact_points *= 1E3 
# save the contact points to a csv file 
contact_df = pd.DataFrame(contact_points, columns=['x','y','z']) 

# save the dataframe 
contact_df.to_pickle(os.path.join(dir_save, output_file.removesuffix(".csv") + ".pkl")) 
contact_df.to_csv(os.path.join(dir_save, output_file), index=False)  

print(f"Pose boundary data saved to {os.path.join(dir_save, output_file)}.") 