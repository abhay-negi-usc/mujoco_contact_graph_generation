from typing import Tuple
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray
from numba import jit, njit, prange

PoseArray = NDArray[np.float64]
TransformMatrix = NDArray[np.float64]


def pose2transform(pose: PoseArray) -> TransformMatrix:
    """Convert 6D/7D pose to 4x4 transformation matrix.

    Args:
        pose: [x,y,z,r,p,y] or [x,y,z,qw,qx,qy,qz]
    Returns:
        4x4 transformation matrix
    """
    if pose.ndim == 1:
        T = np.eye(4)
        if pose.shape[0] == 6:  # Euler angles
            T[:3, :3] = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
        elif pose.shape[0] == 7:  # Quaternion
            T[:3, :3] = R.from_quat(pose[3:], scalar_first=True).as_matrix()
        T[:3, 3] = pose[:3]
        return T

    T = np.zeros((pose.shape[0], 4, 4))
    if pose.shape[1] == 6:  # Euler angles
        for i in range(pose.shape[0]):
            T[i, :3, :3] = R.from_euler("xyz", pose[i, 3:], degrees=True).as_matrix()
            T[i, :3, 3] = pose[i, :3]
    elif pose.shape[1] == 7:  # Quaternion
        for i in range(pose.shape[0]):
            T[i, :3, :3] = R.from_quat(pose[i, 3:], scalar_first=True).as_matrix()
            T[i, :3, 3] = pose[i, :3]
    T[:, 3, 3] = 1
    return T


def poses7D2poses6D(poses: PoseArray) -> PoseArray:
    """Convert quaternion poses to euler angle poses.

    Args:
        poses: [..., x,y,z,qw,qx,qy,qz]
    Returns:
        [..., x,y,z,rx,ry,rz]
    """
    if poses.ndim == 1:
        poses6D = np.zeros(6)
        poses6D[:3] = poses[:3]
        poses6D[3:] = R.from_quat(poses[3:], scalar_first=True).as_euler(
            "xyz", degrees=True
        )
        return poses6D

    poses6D = np.zeros((poses.shape[0], 6))
    for i in range(poses.shape[0]):
        poses6D[i, :3] = poses[i, :3]
        poses6D[i, 3:] = R.from_quat(poses[i, 3:], scalar_first=True).as_euler(
            "xyz", degrees=True
        )
    return poses6D


def transform2pose(T: TransformMatrix) -> PoseArray:
    """Convert 4x4 transformation matrix to 7D pose.

    Args:
        T: 4x4 transformation matrix
    Returns:
        [x,y,z,qw,qx,qy,qz]
    """
    pose = np.zeros(7)
    pose[:3] = T[:3, 3]
    pose[3:] = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    return pose


def pose6D2transform(pose: PoseArray) -> TransformMatrix:
    """Convert 6D euler pose to 4x4 transformation matrix.

    Args:
        pose: [x,y,z,rx,ry,rz]
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def transform2pose6D(T: TransformMatrix) -> PoseArray:
    """Convert 4x4 transformation matrix to 6D euler pose.

    Args:
        T: 4x4 transformation matrix
    Returns:
        [x,y,z,rx,ry,rz]
    """
    pose = np.zeros(6)
    pose[:3] = T[:3, 3]
    pose[3:] = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return pose


def transform_poses(poses: PoseArray, delta_pose: PoseArray) -> PoseArray:
    """Transform array of 7D poses by a given pose.

    Args:
        poses: Array of [x,y,z,qw,qx,qy,qz]
        delta_pose: Transform pose [x,y,z,qw,qx,qy,qz]
    Returns:
        Transformed poses
    """
    T_delta = pose2transform(delta_pose)
    transformed_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        transformed_poses[i] = transform2pose(
            np.linalg.inv(T_delta) @ pose2transform(poses[i])
        )
    return transformed_poses


def transform_poses6D(poses: PoseArray, delta_pose: PoseArray) -> PoseArray:
    """Transform array of 6D poses by a given pose.

    Args:
        poses: Array of [x,y,z,rx,ry,rz]
        delta_pose: Transform pose [x,y,z,rx,ry,rz]
    Returns:
        Transformed poses
    """
    T_delta = pose6D2transform(delta_pose)
    transformed_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        transformed_poses[i] = transform2pose6D(
            np.linalg.inv(T_delta) @ pose6D2transform(poses[i])
        )
    return transformed_poses


def nearest_neighbor(P_t: PoseArray, P_s: PoseArray) -> Tuple[PoseArray, PoseArray]:
    """Find nearest neighbors between two point sets.

    Args:
        P_t: Target points
        P_s: Source points
    Returns:
        Tuple of (correspondences, indices)
    """
    kdtree = KDTree(P_t)
    dist, idx = kdtree.query(P_s)
    correspondences = P_t[idx]
    return correspondences, idx


def compute_delta_pose(P_s: PoseArray, P_t: PoseArray) -> PoseArray:
    """Compute delta pose between source and target pose sets.

    Args:
        P_s: Source poses [N, 6]
        P_t: Target poses [N, 6]
    Returns:
        Delta pose [6]
    """
    mean_pose_s = np.mean(P_s, axis=0) if P_s.ndim > 1 else P_s
    mean_pose_t = np.mean(P_t, axis=0) if P_t.ndim > 1 else P_t
    T_s = pose6D2transform(mean_pose_s)
    T_t = pose6D2transform(mean_pose_t)
    return transform2pose6D(np.linalg.inv(T_t) @ T_s)


def compute_delta_pose_apriori_P_t(
    P_s: PoseArray, T_t_inv: TransformMatrix
) -> PoseArray:
    """Compute delta pose with known inverse target transform.

    Args:
        P_s: Source poses [N, 6]
        T_t_inv: Inverse target transform [4, 4]
    Returns:
        Delta pose [6]
    """
    mean_pose_s = np.mean(P_s, axis=0)
    T_s = pose6D2transform(mean_pose_s)
    return transform2pose6D(T_t_inv @ T_s)


def apply_delta_pose(poses: PoseArray, delta_pose: PoseArray) -> PoseArray:
    """Apply delta pose to array of poses.

    Args:
        poses: Input poses [..., 6]
        delta_pose: Delta pose [..., 6]
    Returns:
        Transformed poses
    """
    if poses.ndim > 1 and delta_pose.ndim == 1:
        poses_new = np.zeros_like(poses)
        for i in range(poses.shape[0]):
            poses_new[i] = transform2pose6D(
                pose6D2transform(poses[i]) @ pose6D2transform(delta_pose)
            )
    elif delta_pose.ndim > 1 and poses.ndim == 1:
        poses_new = np.zeros_like(delta_pose)
        T = pose6D2transform(poses)
        for i in range(delta_pose.shape[0]):
            poses_new[i] = transform2pose6D(T @ pose6D2transform(delta_pose[i]))
    elif delta_pose.ndim > 1 and poses.ndim > 1:
        poses_new = np.zeros_like(poses)
        for i in range(poses.shape[0]):
            poses_new[i] = transform2pose6D(
                pose6D2transform(poses[i]) @ pose6D2transform(delta_pose[i])
            )
    elif delta_pose.ndim == 1 and poses.ndim == 1:
        poses_new = transform2pose6D(
            pose6D2transform(poses) @ pose6D2transform(delta_pose)
        )
    else:
        raise ValueError("Incompatible dimensions")
    return poses_new


def invert_pose6D(poses: PoseArray) -> PoseArray:
    """Invert 6D poses.

    Args:
        poses: Input poses [..., 6]
    Returns:
        Inverted poses
    """
    if poses.ndim == 1:
        return transform2pose6D(np.linalg.inv(pose6D2transform(poses)))

    poses_inv = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        poses_inv[i] = transform2pose6D(np.linalg.inv(pose6D2transform(poses[i])))
    return poses_inv


# Core matrix operations


@njit(fastmath=True)
def ensure_2d(arr):
    """Ensure array is 2D"""
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


@njit(fastmath=True)
def create_transform_matrix():
    """Create a 4x4 identity matrix."""
    T = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        T[i, i] = 1.0
    return T


@njit(fastmath=True)
def pose6D2transform_fast(pose):
    """Optimized 6D pose to transform conversion."""
    T = create_transform_matrix()
    T[0, 3] = pose[0]
    T[1, 3] = pose[1]
    T[2, 3] = pose[2]
    return T


@njit(fastmath=True)
def transform2pose6D_fast(T):
    """Optimized transform to 6D pose conversion."""
    pose = np.zeros(6, dtype=np.float64)
    pose[0] = T[0, 3]
    pose[1] = T[1, 3]
    pose[2] = T[2, 3]
    return pose


@njit
def apply_delta_pose_batch(poses, delta_pose):
    """Optimized batch delta pose application."""
    poses = ensure_2d(poses)
    delta_pose = ensure_2d(delta_pose)

    n_poses = poses.shape[0]
    poses_new = np.zeros((n_poses, 6), dtype=np.float64)

    T2 = pose6D2transform_fast(delta_pose[0])

    for i in prange(n_poses):
        T1 = pose6D2transform_fast(poses[i])
        T_result = np.dot(T1, T2)
        poses_new[i] = transform2pose6D_fast(T_result)

    return poses_new


@njit
def invert_pose6D_batch(poses):
    """Optimized batch pose inversion."""
    poses = ensure_2d(poses)
    n_poses = poses.shape[0]
    poses_inv = np.zeros((n_poses, 6), dtype=np.float64)

    for i in prange(n_poses):
        T = pose6D2transform_fast(poses[i])
        T_inv = np.linalg.inv(T)
        poses_inv[i] = transform2pose6D_fast(T_inv)

    return poses_inv


@njit
def compute_mean_pose(poses):
    """Optimized pose averaging."""
    poses = ensure_2d(poses)
    mean_pose = np.zeros(6, dtype=np.float64)
    n_poses = poses.shape[0]

    for i in range(6):
        sum_val = 0.0
        for j in range(n_poses):
            sum_val += poses[j, i]
        mean_pose[i] = sum_val / float(n_poses)

    return mean_pose


@njit
def compute_transformation_update(T_r_t, T_R_t_correspondences, T_r_S):
    """Optimized transformation update computation."""
    # Ensure all inputs are 2D arrays
    T_r_t = ensure_2d(T_r_t)
    T_R_t_correspondences = ensure_2d(T_R_t_correspondences)
    T_r_S = ensure_2d(T_r_S)

    # Compute transformation update
    T_t_r = invert_pose6D_batch(T_r_t)
    T_R_r = apply_delta_pose_batch(T_R_t_correspondences, T_t_r[0])
    T_R_r_mean = compute_mean_pose(T_R_r)

    # Apply transformations
    T_r_t_new = apply_delta_pose_batch(T_r_t, T_R_r_mean.reshape(1, -1))
    T_r_S_new = apply_delta_pose_batch(T_r_S, T_R_r_mean.reshape(1, -1))

    return T_r_t_new[0], T_r_S_new[0], T_R_r_mean


@njit(fastmath=True)
def compute_mean_pose_fast(poses):
    """Compute mean of poses with explicit loops."""
    if poses.ndim == 1:
        return poses

    mean_pose = np.zeros(6, dtype=np.float64)
    n_poses = poses.shape[0]

    for i in range(6):
        sum_val = 0.0
        for j in range(n_poses):
            sum_val += poses[j, i]
        mean_pose[i] = sum_val / float(n_poses)

    return mean_pose


@njit(fastmath=True)
def compute_delta_pose_fast(P_s: PoseArray, P_t: PoseArray) -> PoseArray:
    """Compute delta pose between source and target pose sets.
    Args:
        P_s: Source poses [N, 6]
        P_t: Target poses [N, 6]
    Returns:
        Delta pose [6]
    """
    # Compute mean poses
    mean_pose_s = compute_mean_pose_fast(P_s)
    mean_pose_t = compute_mean_pose_fast(P_t)

    # Convert to transforms
    T_s = pose6D2transform_fast(mean_pose_s)
    T_t = pose6D2transform_fast(mean_pose_t)

    # Compute inverse and multiplication
    T_t_inv = np.linalg.inv(T_t)
    T_delta = np.dot(T_t_inv, T_s)

    # Convert back to pose
    return transform2pose6D_fast(T_delta)
