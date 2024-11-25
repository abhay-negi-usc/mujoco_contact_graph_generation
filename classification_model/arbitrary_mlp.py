import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, flag_classification=False):
        super(MLP, self).__init__()
        
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
        self.flag_classification = flag_classification 

    def forward(self, x):
        y = self.model(x)
        if self.flag_classification: 
            z = torch.softmax(y, dim=1) 
            return z 
        else: 
            return y 
    
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0) 
