import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Dict
import torch


class BaseContactStateDataset(Dataset):
    """Base class for contact state datasets with caching."""
    
    # Class variable to cache data across instances
    _data_cache: Dict[str, pd.DataFrame] = {}
    
    def __init__(self, data_path: str, filter_out_no_contact: bool = True, seed: int = 42):
        """
        Initialize the base dataset with caching.
        
        Args:
            data_path: Path to the CSV data file
            filter_out_no_contact: If True, filters out rows where contact_type is -1
            seed: Random seed for data shuffling
        """
        self.data_path = data_path
        self.filter_out_no_contact = filter_out_no_contact
        self.seed = seed
        
        # Initialize peg-hole class combinations
        self.hole_classes = ['HF1', 'HF2', 'HE1', 'HE2', 'HE3', 'HE4', 'HE5', 'HE6',
                           'HV1', 'HV2', 'HV3', 'HV4']
        self.peg_classes = ['PF1', 'PF2', 'PF3', 'PF4', 'PE1', 'PE2', 'PE3', 'PE4']
        self.peg_hole_classes = [f'{h}-{p}' for h in self.hole_classes 
                                            for p in self.peg_classes]
        
        # Load or get cached data
        self._load_data()
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        
        # Store processed features and labels as tensors
        self.features = None
        self.labels = None
    
    def _load_data(self) -> None:
        """Load data from cache or file and process it."""
        cache_key = f"{self.data_path}_{self.filter_out_no_contact}_{self.seed}"
        
        if cache_key not in self._data_cache:
            # Read data if not in cache
            self.data = pd.read_csv(self.data_path)
            if self.filter_out_no_contact:
                self.data = self.data[self.data["contact_type"] != -1]
            # Store in cache
            self._data_cache[cache_key] = self.data
        else:
            # Use cached data
            self.data = self._data_cache[cache_key]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features) if self.features is not None else 0
    
    def _validate_index(self, idx: int) -> None:
        """Validate that the index is within bounds."""
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")


class Pose2ContactStateDataset(BaseContactStateDataset):
    """Dataset for pose to contact state mapping"""
    
    def __init__(self, data_path: str, filter_out_no_contact: bool = True, seed: int = 42):
        super().__init__(data_path, filter_out_no_contact, seed)  # superclass
        
        self.pose_column_headings = ['X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW']
        
        # Process data once during initialization
        self._process_data()
    
    def _process_data(self) -> None:

        # Select relevant columns
        processed_data = self.data[self.pose_column_headings + self.peg_hole_classes]
        processed_data = processed_data.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(
            processed_data[self.pose_column_headings]
        )
        
        # Convert to tensors
        self.features = torch.FloatTensor(scaled_features)
        self.labels = torch.FloatTensor(processed_data[self.peg_hole_classes].values)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        self._validate_index(idx)
        return self.features[idx], self.labels[idx]


class Wrench2ContactStateDataset(BaseContactStateDataset):
    """Dataset for wrench to contact state mapping"""
    
    def __init__(self, data_path: str, filter_out_no_contact: bool = True, seed: int = 42):

        super().__init__(data_path, filter_out_no_contact, seed)  # superclass
        
        # Define column headings
        self.wrench_column_headings = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']
        self.normalized_wrench_column_headings = ['fx', 'fy', 'tx', 'ty', 'tz']
        
        # Process data once during initialization
        self._process_data()
    
    def _process_data(self) -> None:

        # Calculate normalized features
        for axis in ['x', 'y']:
            self.data[f'f{axis}'] = self.data[f'F{axis.upper()}'] / self.data['FZ']
            self.data[f't{axis}'] = self.data[f'T{axis.upper()}'] / self.data['FZ']
        self.data['tz'] = self.data['TZ'] / self.data['FZ']
        
        # Filter for non-zero classes
        self.peg_hole_classes_nonzero = [
            cls for cls in self.peg_hole_classes 
            if self.data[cls].sum() > 0
        ]
        
        # Select relevant columns
        input_columns = self.wrench_column_headings + self.normalized_wrench_column_headings
        
        # Scale features
        scaled_wrench = self.scaler.fit_transform(
            self.data[self.wrench_column_headings]
        )
        scaled_normalized = self.scaler.fit_transform(
            self.data[self.normalized_wrench_column_headings]
        )
        
        # Combine scaled features and convert to tensors
        self.features = torch.FloatTensor(np.hstack([scaled_wrench, scaled_normalized]))
        self.labels = torch.FloatTensor(self.data[self.peg_hole_classes_nonzero].values)
        
        # Shuffle data
        indices = torch.randperm(len(self.features))
        self.features = self.features[indices]
        self.labels = self.labels[indices]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        self._validate_index(idx)
        return self.features[idx], self.labels[idx]