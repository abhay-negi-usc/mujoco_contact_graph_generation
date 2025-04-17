import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle
import matplotlib.pyplot as plt

# read in data 
num_sets = 10 
pose_labels = ['x', 'y', 'z', 'a', 'b', 'c'] 
dir_data = "/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/extrusion_data/perturb_v3/processed_data" 
tolerances = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]  
tolerances = sorted(tolerances, reverse=True) 
geometry = "extrusion"

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

print(f"Length of data: {len(df)}")
print(f"Number of contact poses: {len(poses_contact_all)}")
print(f"Number of freespace poses: {len(poses_freespace_all)}")

# create kd tree for contact poses 
print("Creating KDTree for contact poses")
kdtree_freespace = KDTree(poses_freespace_all, leaf_size=30, metric='euclidean') 
print("KDTree created for freespace poses") 

num_points_list = []
# create list of empty lists for each tolerance 
poses_boundary_list = [[] for _ in tolerances] 
poses_boundary_dist_list = [[] for _ in tolerances] 

# loop through contact poses 
for i, pose in enumerate(poses_contact_all): 
    # find nearest neighbor in contact and freespace 
    dist, ind = kdtree_freespace.query([pose], k=1) 

    for j, tolerance in enumerate(tolerances): 
        # if distance is less than tolerance, add to boundary poses
        if dist < tolerance: 
            poses_boundary_list[j].append(pose) 
            poses_boundary_dist_list[j].append(dist) 
        else: 
            # if criteria is not met for greater tolerance value, then break because lower tolerance will also not be met 
            break 

    # print progress every 1% 
    if i % (len(poses_contact_all)//100) == 0: 
        print(f"Progress: {i/len(poses_contact_all)*100:.2f}%") 

for j, tolerance in enumerate(tolerances): 
    # convert to np array 
    poses_boundary = np.array(poses_boundary_list[j])
    poses_boundary_dist = np.array(poses_boundary_dist_list[j]) 

    num_points_list.append(len(poses_boundary)) 

    print(f"Number of boundary poses: {num_points_list[j]}") 

    # save to csv and pkl 
    df_boundary = pd.DataFrame(poses_boundary, columns=pose_labels) 
    df_boundary['dist'] = np.squeeze(poses_boundary_dist, axis=-1) 
    df_boundary.to_csv(f"{dir_data}/{geometry}_refined_boundary_poses_tol_{tolerance}_num_pts_{num_points_list[j]}.csv", index=False) 
    df_boundary.to_pickle(f"{dir_data}/{geometry}_refined_boundary_poses_tol_{tolerance}_num_pts_{num_points_list[j]}.pkl") 

plt.figure() 
plt.scatter(tolerances, num_points_list)
plt.xlabel("Tolerance (mm and degrees)") 
plt.ylabel("Number of Points")
plt.title("Number of Contact Points within Tolerance of Freespace Points") 
plt.grid(True) 
# save plot 
plt.savefig(f"{dir_data}/num_points_vs_tol.png") 


