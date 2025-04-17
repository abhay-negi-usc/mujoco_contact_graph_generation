import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle

# read in data
geometry = "extrusion" # "extrusion" "cross" "plug_3_pin" 
num_sets = 10 
pose_labels = ['x', 'y', 'z', 'a', 'b', 'c'] 
dir_data = "/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/extrusion_data/perturb_v3/processed_data" 
poses_contact_all = [] 
poses_freespace_all = [] 
for set_num in range(num_sets): 
    df = pd.read_pickle(f"{dir_data}/all_sim_state_data_{set_num}.pkl") 
    poses_contact = df[df['contact']>0][pose_labels].values
    poses_freespace = df[df['contact']==0][pose_labels].values
    poses_contact_all.append(poses_contact)
    poses_freespace_all.append(poses_freespace) 
    print("Read file ", set_num)

poses_contact_all = np.vstack(poses_contact_all)
poses_freespace_all = np.vstack(poses_freespace_all)

tolerance = 0.05

print(f"Length of data: {len(df)}")

print(f"Number of contact poses: {len(poses_contact_all)}")
print(f"Number of freespace poses: {len(poses_freespace_all)}")

# create kd tree for contact poses 
print("Creating KDTree for contact poses")
kdtree_freespace = KDTree(poses_freespace_all, leaf_size=30, metric='euclidean') 
print("KDTree created for freespace poses") 

poses_boundary = [] 
poses_boundary_dist = [] 
# loop through contact poses 
for i, pose in enumerate(poses_contact_all): 
    # find nearest neighbor in contact and freespace 
    dist, ind = kdtree_freespace.query([pose], k=1) 

    # if distance is less than tolerance, add to boundary poses
    if dist < tolerance: 
        poses_boundary.append(pose) 
        poses_boundary_dist.append(dist) 

    # print progress every 1% 
    if i % (len(poses_contact_all)//100) == 0: 
        print(f"Progress: {i/len(poses_contact_all)*100:.2f}%") 

# convert to np array 
poses_boundary = np.array(poses_boundary)
poses_boundary_dist = np.array(poses_boundary_dist) 

print(f"Number of boundary poses: {len(poses_boundary)}") 

# save to csv and pkl 
df_boundary = pd.DataFrame(poses_boundary, columns=pose_labels) 
df_boundary['dist'] = np.squeeze(poses_boundary_dist, axis=-1) 
df_boundary.to_csv(f"{dir_data}/{geometry}_refined_boundary_poses_tol_{tolerance}_num_pts_{len(poses_boundary)}.csv", index=False) 
df_boundary.to_pickle(f"{dir_data}/{geometry}_refined_boundary_poses_tol_{tolerance}_num_pts_{len(poses_boundary)}.pkl") 