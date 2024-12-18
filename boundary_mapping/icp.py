from typing import List, Optional
import numpy as np

from numpy.typing import NDArray
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        """Plot 3D visualization of pose map colored by rotation components."""
        x, y, z = self.map[:, 0], self.map[:, 1], self.map[:, 2]
        a, b, c = self.map[:, 3], self.map[:, 4], self.map[:, 5]

        fig = plt.figure()
        point_size = 1.0

        # Create three 3D subplots for different rotation components
        ax0 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2 = fig.add_subplot(2, 1, 2, projection="3d")

        # Plot rotation components with different colors
        self._add_subplot_scatter(ax0, x, y, z, a, "a", point_size)
        self._add_subplot_scatter(ax1, x, y, z, b, "b", point_size)
        self._add_subplot_scatter(ax2, x, y, z, c, "c", point_size)

        def on_plot_move(event: plt.MouseEvent) -> None:
            """Synchronize view angles across all subplots."""
            source_ax = event.inaxes
            if source_ax in [ax0, ax1, ax2]:
                axes = [ax0, ax1, ax2]
                for ax in axes:
                    if ax != source_ax:
                        ax.view_init(elev=source_ax.elev, azim=source_ax.azim)
                        ax.set_xlim(source_ax.get_xlim())
                        ax.set_ylim(source_ax.get_ylim())
                        ax.set_zlim(source_ax.get_zlim())
                plt.draw()

        fig.canvas.mpl_connect("motion_notify_event", on_plot_move)
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
        """Run ICP algorithm for a single initial guess.

        Args:
            guess_idx: Index of current guess
            max_iter: Maximum iterations
            tol: Convergence tolerance
            flag_iteration_output: Whether to print progress
        """
        T_r_S = self.initial_guesses[guess_idx]
        T_r_t = apply_delta_pose(T_r_S, self.observation_clean)

        res, err, MAE, max_err = [], [], [], []

        for i in range(max_iter):
            # Find nearest neighbors and compute transformation
            dist, idx = self.kdtree.query(T_r_t)
            T_R_t_correspondences = self.map[idx]

            # Update estimates and compute errors
            T_t_r = invert_pose6D(T_r_t)
            T_R_r = apply_delta_pose(T_R_t_correspondences, T_t_r)
            T_R_r_mean = np.mean(T_R_r, axis=0)

            # Store metrics
            res.append(np.mean(np.abs(T_R_r_mean)))
            err.append(compute_delta_pose(T_r_S, self.true_transform))
            MAE.append(np.mean(np.abs(err[-1])))
            max_err.append(np.max(np.abs(err[-1])))

            if res[-1] < tol:
                if flag_iteration_output:
                    print(f"Converged at iteration {i}, residual: {res[-1]}")
                break

            # Update poses for next iteration
            T_r_t = apply_delta_pose(T_R_r_mean, T_r_t)
            T_r_S = apply_delta_pose(T_R_r_mean, T_r_S)

        # Store results for this guess
        self.transform_estimates[guess_idx] = T_r_S
        self.transform_errors[guess_idx] = err[-1]
        self.residuals.append(res)
        self.errors.append(err)
        self.MAE.append(MAE)
        self.max_err.append(max_err)

        if flag_iteration_output:
            print(
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
