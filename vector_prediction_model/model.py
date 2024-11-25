import torch 
import torch.nn as nn

class BinaryVectorPredictor(nn.Module):
    def __init__(self, layer_sizes):
        super(BinaryVectorPredictor, self).__init__()
        
        # Create layers based on the input layer sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2: 
                layers.append(nn.ReLU())  
        # consolidate layers into model 
        self.model = nn.Sequential(*layers) 
        # initialize weights 
        self.init_weights() 

        # Sigmoid to ensure the output is between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.model(x)
        z = self.sigmoid(y) 
        return z 
    
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0) 
