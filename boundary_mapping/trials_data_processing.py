import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R 
import pickle 
import os 
import glob 
from mpl_toolkits.mplot3d import Axes3D 
import random 
import heapq

geometry = "square_B" 
dir_results = f"/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/{geometry}_data/perturb_v3/" 
dir_pkl = dir_results + "/pkl" 
pkl_files_all = sorted(glob.glob(os.path.join(dir_pkl, "*.pkl")), key=os.path.getmtime)

dir_save = dir_pkl.removesuffix("pkl") + "processed_data"
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

# list of all contact state history 
N_timesteps = 500 
N_trials = 100_000 
N_trials = len(pkl_files_all) if len(pkl_files_all) < N_trials else N_trials
# set_num = 1

for set_num in range(1000): 

    print(f"\n\nProcessing set {set_num}")

    pkl_files = pkl_files_all[N_trials*(set_num):N_trials*(set_num+1)] 
    pose_boundary_list = [] 

    mujoco_state_labels = ['time','X','Y','Z','QW','QX','QY','QZ','X_dot','Y_dot','Z_dot','A_dot','B_dot','C_dot','FX','FY','FZ','TX','TY','TZ','x','y','z','qw','qx','qy','qz','contact','quasi_static'] 
    N_state_mujoco = len(mujoco_state_labels) 
    N_states = N_state_mujoco 
    state_hist_all = np.zeros((N_trials, N_timesteps, N_states)) 
    quasi_static_hist_all = np.zeros((N_timesteps, N_trials), dtype=bool) 
    quasi_static_thresh = 0.001 # 1 mm/s 
    flag_quasi_static_constraint = False 

    for i, pkl_file in enumerate(pkl_files): 

        # Read the pickle file
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # unpack data 
        state_hist = data['state_hist'] 
        contact_num = data['contact_num'] 
        contact_geom1 = data['contact_geom1'] 
        contact_geom2 = data['contact_geom2'] 
        # contact_dist = data['contact_dist'] 
        contact_pos = data['contact_pos'] 
        # contact_frame = data['contact_frame'] 
        # ctrl_hist = data['ctrl_hist'] 
        sensor_hist = data['sensor_hist']   

        contact_num = np.array(contact_num).reshape(N_timesteps,1) 

        # state_hist_all[:,:,i] = np.hstack((state_hist, sensor_hist, contact_num))  
        quasi_static = np.zeros((N_timesteps,1)) 
        for j in range(N_timesteps): # iterate through each time step 

            # check if the contact is quasi-static 
            if np.max(np.abs(state_hist[j,8:11])) < quasi_static_thresh: 
                quasi_static[j,0] = 1 
            else:  
                quasi_static[j,0] = 0 

        # combine state_hist, sensor_hist, contact_num, and quasi_static into state_hist_all 
        state_hist_all[i,:,:] = np.hstack((state_hist, sensor_hist, contact_num, quasi_static)) 

        # print progress every 10% 
        if i % (N_trials//10) == 0: 
            print(f"Processed {i} trials out of {N_trials}") 

    state_hist_all_flat = state_hist_all.reshape(N_timesteps*N_trials, N_states)
    df_state = pd.DataFrame(state_hist_all_flat, columns=mujoco_state_labels)    
    
    # remove nans from df_state and reindex 
    df_state = df_state.dropna().reset_index(drop=True) 

    # compute a,b,c and convert x,y,z to mm 

    df_state[['x', 'y', 'z']] *= 1000 
    for index, row in df_state.iterrows(): 
        qnorm = np.linalg.norm([row['QX'], row['QY'], row['QZ'], row['QW']]) 
        if qnorm < 1e-6: 
            import pdb; pdb.set_trace() 
        c,b,a = R.from_quat([row['QX'], row['QY'], row['QZ'], row['QW']]).as_euler('xyz', degrees=True) 
        df_state.at[index, 'a'] = a 
        df_state.at[index, 'b'] = b 
        df_state.at[index, 'c'] = c 

        # print progress every 10% 
        if index % (N_trials*N_timesteps//10) == 0: 
            print(f"Processed {index} trials out of {N_trials*N_timesteps}")

    # save the processed data 
    df_state.to_csv(os.path.join(dir_save, f"all_sim_state_data_{set_num}.csv"), index=False) 
    # save as pkl 
    df_state.to_pickle(os.path.join(dir_save, f"all_sim_state_data_{set_num}.pkl")) 
