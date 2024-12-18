from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.spatial import KDTree
from loguru import logger

from boundary_mapping.util import *
from boundary_mapping.icp import ContactPoseMap


@dataclass
class TestResults:
    """Container for test results."""

    initial_contacts: NDArray[np.float64]
    transform_errors: NDArray[np.float64]
    mean_absolute_error: NDArray[np.float64]
    std_deviation_error: NDArray[np.float64]


class ContactPoseMapTester:
    """Testing framework for ContactPoseMap implementation."""

    def __init__(
        self,
        map_data: NDArray[np.float64],
        n_downsampled_poses: int = 100_000,
        n_test_iterations: int = 10,
        n_observations: int = 1000,
        n_icp_guesses: int = 10,
        observation_noise_std: float = 1.0,
        perturbation_std: float = 1.0,
        true_transform_std: float = 5.0,
        initial_guess_uncertainty: float = 3.0,
    ):
        """Initialize tester with configuration parameters.

        Args:
            map_data: Input pose map data
            n_downsampled_poses: Number of poses after downsampling
            n_test_iterations: Number of test iterations
            n_observations: Number of observations per test
            n_icp_guesses: Number of ICP initial guesses
            observation_noise_std: Standard deviation of observation noise
            perturbation_std: Standard deviation for pose perturbations
            true_transform_std: Standard deviation for true transform sampling
            initial_guess_uncertainty: Uncertainty in initial guess
        """
        self.map_data = map_data
        self.n_downsampled_poses = n_downsampled_poses
        self.n_test_iterations = n_test_iterations
        self.n_observations = n_observations
        self.n_icp_guesses = n_icp_guesses
        self.observation_noise_std = observation_noise_std
        self.perturbation_std = perturbation_std
        self.true_transform_std = true_transform_std
        self.initial_guess_uncertainty = initial_guess_uncertainty

        self.contact_map: Optional[ContactPoseMap] = None

    def generate_test_observation(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Generate a single test observation.

        Returns:
            Tuple containing:
            - Initial contact pose
            - Map sample without observation
            - True transform
            - Observation data
        """
        # Sample initial contact pose
        idx_observation_center = np.random.choice(len(self.map_data), 1, replace=False)
        initial_contact_pose = self.map_data[idx_observation_center]

        # Generate perturbed poses
        perturbations = initial_contact_pose + np.random.normal(
            0, self.perturbation_std, (self.n_observations, 6)
        )

        # Find nearest neighbors in map
        local_poses_map_sample, idx_neighbors = self._find_nearest_neighbors(
            self.map_data, perturbations
        )

        # Combine indices and remove from map
        idx_observations = np.append(idx_neighbors, idx_observation_center)
        map_data_del_obs = np.delete(self.map_data, idx_observations, axis=0)

        # Generate true transform and observation
        true_transform = np.random.normal(0, self.true_transform_std, 6)
        true_transform_inverse = invert_pose6D(true_transform)
        observation = apply_delta_pose(true_transform_inverse, local_poses_map_sample)

        return initial_contact_pose, map_data_del_obs, true_transform, observation

    def _find_nearest_neighbors(
        self, reference_poses: NDArray[np.float64], query_poses: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find nearest neighbors using KDTree.

        Args:
            reference_poses: Reference pose set
            query_poses: Query pose set

        Returns:
            Tuple of (nearest poses, indices)
        """
        kdtree = KDTree(reference_poses)
        dist, idx = kdtree.query(query_poses)
        return reference_poses[idx], idx

    def run_tests(self) -> TestResults:
        """Run complete test suite.

        Returns:
            TestResults containing error statistics
        """
        initial_contact_list: List[NDArray[np.float64]] = []
        error_list: List[NDArray[np.float64]] = []

        for i in range(self.n_test_iterations):
            # Generate test data
            initial_contact_pose, map_data_del_obs, true_transform, observation = (
                self.generate_test_observation()
            )

            # Initialize or update contact map
            if i == 0:
                self.contact_map = ContactPoseMap(map_data_del_obs)
                self.contact_map.downsample_map(self.n_downsampled_poses)

            # Configure and run ICP
            self.contact_map.set_observation(observation, true_transform)
            self.contact_map.add_noise_to_observation(self.observation_noise_std)
            self.contact_map.set_initial_guesses(
                true_transform, self.initial_guess_uncertainty, self.n_icp_guesses
            )

            # Run ICP and collect results
            self.contact_map.run_ICP(
                max_iter=50,
                tol=1e-18,
                flag_iteration_output=False,
                flag_plot_results=False,
            )

            logger.debug(
                f"Iteration {i+1}/{self.n_test_iterations} - "
                f"Final Transform Error: {self.contact_map.transform_final_error}"
            )

            initial_contact_list.append(initial_contact_pose)
            error_list.append(self.contact_map.transform_final_error)

        # Compute error statistics
        error_matrix = np.array(error_list)
        initial_contacts = np.squeeze(np.array(initial_contact_list))
        mean_absolute_error = np.mean(np.abs(error_matrix), axis=0)
        std_deviation_error = np.std(error_matrix, axis=0)

        return TestResults(
            initial_contacts=initial_contacts,
            transform_errors=error_matrix,
            mean_absolute_error=mean_absolute_error,
            std_deviation_error=std_deviation_error,
        )

    def visualize_results(self, results: TestResults) -> None:
        """Visualize test results.

        Args:
            results: TestResults object containing test statistics
        """
        if self.contact_map is not None:
            self.contact_map.plot_map()

        logger.debug("\nTest Results Summary:")
        logger.debugf("Mean Absolute Error: {results.mean_absolute_error}")
        logger.debugf("Standard Deviation of Error: {results.std_deviation_error}")

        # Additional visualizations could be added here


if __name__ == "__main__":

    # TODO: Input path to dataset
    input_path = "/home/dhanush/mujoco_contact_graph_generation/contact_data/cross_peg_contact_mapping_real.csv"

    # Real dataset
    real_df = pd.read_csv(input_path)
    real_df.rename(
        columns={"X": "x", "Y": "y", "Z": "z", "A": "a", "B": "b", "C": "c"},
        inplace=True,
    )
    real_df = real_df[real_df["contact"] == True]

    # Initialize tester
    tester = ContactPoseMapTester(
        map_data=real_df[["x", "y", "z", "a", "b", "c"]].values,
        n_downsampled_poses=100_000,
        n_test_iterations=10,
    )

    # Run tests
    results = tester.run_tests()

    # Visualize results
    tester.visualize_results(results)
