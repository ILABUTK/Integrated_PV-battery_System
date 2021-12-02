"""
MLP model, for RL.
"""

# import
import torch
import torch.nn as nn
# import torch.nn.functional as F
from collections import OrderedDict


# MLP class
class Q_MLP(nn.Module):
    """
    MLP for deep Q-learning.
    """
    def __init__(self, hidden_layer_shape, input_size, output_size, seed):
        """
        hidden_layer_shape: list, the number of neurons for every layer;
        input_size: number of states;
        output_size: number of actions;
        seed: random seed.
        """
        super().__init__()
        # parameters
        self.seed = torch.manual_seed(seed)
        self.hidden_layer_shape = hidden_layer_shape
        # NN, adding layers dinamically.
        linear = OrderedDict()
        for i in range(len(self.hidden_layer_shape) + 1):
            # input
            if i == 0:
                linear[str(i)] = nn.Linear(
                    input_size, self.hidden_layer_shape[i]
                )
            # output
            elif i == len(self.hidden_layer_shape):
                linear[str(i)] = nn.Linear(
                    self.hidden_layer_shape[i - 1], output_size
                )
            # ``hidden" layers
            else:
                linear[str(i)] = nn.Linear(
                    self.hidden_layer_shape[i - 1],
                    self.hidden_layer_shape[i]
                )
        self.layers = nn.Sequential(linear)

    def forward(self, input_seq):
        """
        input_seq: states, torch.tensor.
        """
        linear_out = 0
        # make a prediction
        for i in range(len(self.hidden_layer_shape) + 1):
            current_layer = self.layers[i]
            # input
            if i == 0:
                linear_out = torch.sigmoid(current_layer(input_seq))
            # output, no relu
            elif i == len(self.hidden_layer_shape):
                linear_out = current_layer(linear_out)
            # hidden layers
            else:
                linear_out = torch.sigmoid(current_layer(linear_out))
        # returns the prediction
        return linear_out
