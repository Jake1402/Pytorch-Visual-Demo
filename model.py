import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, inputs, nodes, output, activation):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=nodes),
            activation(),
            nn.Linear(in_features=nodes, out_features=output)
        )

        self.activation_func = activation()

    def forward(self, X):
        return self.network(X)
        