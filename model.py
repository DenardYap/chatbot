import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, input_size, output_size):
        """Define the architecture, i.e. what layers our network contains. 
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions."""
        super().__init__()

        self.hidden_size = 8
        self.l1 = nn.Linear(input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""

        torch.manual_seed(69)

        for conv in [self.l1, self.l2, self.l3]:
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        z = F.relu(self.l1(x))
        # dropout
        z = F.relu(self.l2(z))
        z = self.l3(z)

        return z 