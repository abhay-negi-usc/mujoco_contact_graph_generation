from typing import List, Optional
import numpy as np

from numpy.typing import NDArray
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loguru import logger

from boundary_mapping.util import *


class ContactPoseMap:
    """Handles contact pose mapping and ICP-based pose estimation."""

    def __init__(self, pose_data: NDArray[np.float64]):
        """Initialize with pose map data.

        Args:
            pose_data: Nx6 array of poses [x,y,z,rx,ry,rz]
        """
        self.map: NDArray[np.float64] = pose_data
        self.kdtree: KDTree = KDTree(self.map)
        self.observation: Optional[NDArray[np.float64]] = None
        self.observation_clean: Optional[NDArray[np.float64]] = None
        self.initial_guesses: Optional[NDArray[np.float64]] = None
        self.true_transform: Optional[NDArray[np.float64]] = None

        # ICP results storage
        self.transform_estimates: Optional[NDArray[np.float64]] = None
        self.transform_errors: Optional[NDArray[np.float64]] = None
        self.residuals: List[List[float]] = []
        self.errors: List[List[NDArray[np.float64]]] = []
        self.MAE: List[List[float]] = []
        self.max_err: List[List[float]] = []
        self.transform_final_estimate: Optional[NDArray[np.float64]] = None
        self.transform_final_error: Optional[NDArray[np.float64]] = None

    def downsample_map(self, N: int) -> None:
        """Randomly downsample the pose map.

        Args:
            N: Target number of poses after downsampling
        """
        if N > len(self.map):
            raise ValueError("N must be less than the number of poses in the map")
        idx = np.random.choice(len(self.map), N, replace=False)
        self.map = self.map[idx]
        self.kdtree = KDTree(self.map)

    def plot_map(self) -> None:
        """Plot 3D visualization of pose map with three subplots for rotation components."""
        x, y, z = self.map[:, 0], self.map[:, 1], self.map[:, 2]
        a, b, c = self.map[:, 3], self.map[:, 4], self.map[:, 5]

        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 10))

        # First subplot: x,y,z colored by 'a'
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        scatter1 = ax1.scatter(x, y, z, c=a, cmap="viridis", s=1)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Poses colored by rotation A")
        plt.colorbar(scatter1, ax=ax1)

        # Second subplot: x,y,z colored by 'b'
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        scatter2 = ax2.scatter(x, y, z, c=b, cmap="viridis", s=1)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Poses colored by rotation B")
        plt.colorbar(scatter2, ax=ax2)

        # Third subplot: x,y,z colored by 'c'
        ax3 = fig.add_subplot(2, 1, 2, projection="3d")
        scatter3 = ax3.scatter(x, y, z, c=c, cmap="viridis", s=1)
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.set_title("Poses colored by rotation C")
        plt.colorbar(scatter3, ax=ax3)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def _add_subplot_scatter(
        self,
        ax: Axes3D,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        c: NDArray[np.float64],
        label: str,
        size: float,
    ) -> None:
        """Add scatter plot to a subplot with proper formatting.

        Args:
            ax: Matplotlib 3D axes
            x, y, z: Coordinate arrays
            c: Color array
            label: Colorbar label
            size: Point size
        """
        sc = ax.scatter(x, y, z, c=c, cmap="turbo", s=size)
        plt.colorbar(sc, ax=ax).set_label(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def set_observation(
        self, observation: NDArray[np.float64], true_transform: NDArray[np.float64]
    ) -> None:
        """Set observation data and ground truth transform.

        Args:
            observation: Observed poses
            true_transform: Ground truth transformation
        """
        self.observation = observation
        self.observation_clean = observation
        self.true_transform = true_transform

    def downsample_observation(self, N: int) -> None:
        """Randomly downsample the observation data.

        Args:
            N: Target number of poses after downsampling
        """
        if self.observation is None:
            raise ValueError("No observation provided")
        if N > len(self.observation):
            raise ValueError("N must be less than the number of observation")
        idx = np.random.choice(len(self.observation), N, replace=False)
        self.observation = self.observation[idx]
        self.observation_clean = self.observation_clean[idx]

    def set_initial_guesses(
        self, initial_guess: NDArray[np.float64], uncertainty_std: float, N: int
    ) -> None:
        """Generate initial pose guesses with Gaussian noise.

        Args:
            initial_guess: Mean initial guess
            uncertainty_std: Standard deviation for noise
            N: Number of guesses to generate
        """
        self.initial_guess_mean = initial_guess
        self.initial_guesses = np.random.normal(initial_guess, uncertainty_std, (N, 6))

    def add_noise_to_observation(self, noise_std: float) -> None:
        """Add Gaussian noise to observation data.

        Args:
            noise_std: Standard deviation of noise
        """
        if self.observation is None:
            raise ValueError("No observation provided")
        self.observation += np.random.normal(0, noise_std, self.observation.shape)

    def run_ICP(
        self,
        max_iter: int = 100,
        tol: float = 1e-18,
        flag_iteration_output: bool = True,
        flag_plot_results: bool = True,
    ) -> None:
        """Run Iterative Closest Point algorithm with multiple initial guesses.

        Args:
            max_iter: Maximum iterations per guess
            tol: Convergence tolerance
            flag_iteration_output: Whether to print progress
            flag_plot_results: Whether to plot results after completion
        """
        self._validate_icp_inputs()

        N_guesses = self.initial_guesses.shape[0]
        self._initialize_icp_outputs(N_guesses)

        # Frame definitions:
        # R: robot base frame
        # r: estimate of robot base frame
        # S: origin of contact observation frame
        # t: individual contact observation pose
        T_R_t = self.map
        T_t_R = invert_pose6D(T_R_t)

        for j in range(N_guesses):
            # Run ICP for each initial guess
            self._run_single_icp(j, max_iter, tol, flag_iteration_output)

        # Compute final estimates
        self.transform_final_estimate = np.mean(self.transform_estimates, axis=0)
        self.transform_final_error = compute_delta_pose(
            self.transform_final_estimate, self.true_transform
        )

        if flag_plot_results:
            self.plot_results()

    def _validate_icp_inputs(self) -> None:
        """Validate required inputs for ICP algorithm."""
        if self.observation is None:
            raise ValueError("No observation provided")
        if self.initial_guesses is None:
            raise ValueError("No initial guesses provided")
        if self.true_transform is None:
            raise ValueError("No true transform provided")

    def _initialize_icp_outputs(self, N_guesses: int) -> None:
        """Initialize data structures for ICP results.

        Args:
            N_guesses: Number of initial guesses
        """
        self.transform_estimates = np.zeros((N_guesses, 6))
        self.transform_errors = np.zeros((N_guesses, 6))
        self.residuals = []
        self.errors = []
        self.MAE = []
        self.max_err = []

    def _run_single_icp(
        self, guess_idx: int, max_iter: int, tol: float, flag_iteration_output: bool
    ) -> None:
        """Optimized ICP algorithm for a single initial guess."""
        T_r_S = self.initial_guesses[guess_idx]
        T_r_t = apply_delta_pose_batch(T_r_S[None], self.observation_clean)[0]

        res, err, MAE, max_err = [], [], [], []

        for i in range(max_iter):
            # Find nearest neighbors using KDTree (kept outside numba)
            dist, idx = self.kdtree.query(T_r_t)
            T_R_t_correspondences = self.map[idx]

            # Use optimized transformation update
            T_r_t, T_r_S, T_R_r_mean = compute_transformation_update(
                T_r_t, T_R_t_correspondences, T_r_S
            )

            # Compute and store metrics
            residual = np.mean(np.abs(T_R_r_mean))
            error = compute_delta_pose(T_r_S, self.true_transform)
            mae = np.mean(np.abs(error))
            max_error = np.max(np.abs(error))

            res.append(residual)
            err.append(error)
            MAE.append(mae)
            max_err.append(max_error)

            if residual < tol:
                if flag_iteration_output:
                    logger.debug(f"Converged at iteration {i}, residual: {residual}")
                break

        # Store results for this guess
        self.transform_estimates[guess_idx] = T_r_S
        self.transform_errors[guess_idx] = err[-1]
        self.residuals.append(res)
        self.errors.append(err)
        self.MAE.append(MAE)
        self.max_err.append(max_err)

        if flag_iteration_output:
            logger.debug(
                f"Progress: {guess_idx+1}/{len(self.initial_guesses)}, "
                f"Res: {res[-1]}, MAE: {MAE[-1]}, Max Err: {max_err[-1]}"
            )

    def plot_results(self) -> None:
        """Plot ICP results including residuals, errors, and convergence metrics."""
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        # Plot residuals
        for i, residual in enumerate(self.residuals):
            ax[0, 0].plot(np.arange(len(residual)), residual, label=f"{i}")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_title("Residuals")
        ax[0, 0].set_xlabel("Iteration")
        ax[0, 0].set_ylabel("Residual")
        ax[0, 0].grid(True)

        # Plot final vs initial MAE
        for mae_seq in self.MAE:
            ax[0, 1].scatter(mae_seq[0], mae_seq[-1])
        ax[0, 1].set_title("Final vs Initial MAE")
        ax[0, 1].set_xlabel("Initial MAE")
        ax[0, 1].set_ylabel("Final MAE")
        ax[0, 1].grid(True)

        # Plot maximum error
        for i, max_err in enumerate(self.max_err):
            ax[1, 0].plot(np.arange(len(max_err)), max_err, label=f"{i}")
        ax[1, 0].set_title("Max. Absolute Error")
        ax[1, 0].set_xlabel("Iteration")
        ax[1, 0].set_ylabel("Max. Error")
        ax[1, 0].grid(True)

        # Plot MAE
        for i, mae in enumerate(self.MAE):
            ax[1, 1].plot(np.arange(len(mae)), mae, label=f"{i}")
        ax[1, 1].set_title("Mean Absolute Error")
        ax[1, 1].set_xlabel("Iteration")
        ax[1, 1].set_ylabel("MAE")
        ax[1, 1].grid(True)

        plt.tight_layout()
        plt.show()