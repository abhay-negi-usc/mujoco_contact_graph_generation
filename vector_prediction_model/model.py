import torch
import torch.nn as nn
from typing import List, Optional
from torch import Tensor


class BinaryVectorPredictor(nn.Module):


    def __init__(self, layer_sizes: List[int], dropout_rate: Optional[float] = None) -> None:

        super().__init__()
        
        # Input validation
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layer sizes are required")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers")

        # Build network architecture
        self.model = self._build_network(layer_sizes, dropout_rate)
        
        # Initialize weights
        self._init_weights()

    def _build_network(self, layer_sizes: List[int], dropout_rate: Optional[float]) -> nn.Sequential:

        layers = []
        
        # Create layers with activations and dropout
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.ReLU() if out_size != layer_sizes[-1] else nn.Sigmoid()
            ])
            
            # Add dropout between hidden layers if asked.
            if dropout_rate and out_size != layer_sizes[-1]:
                layers.append(nn.Dropout(p=dropout_rate))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        with torch.no_grad():
            outputs = self.forward(x)
            return (outputs >= threshold).float()