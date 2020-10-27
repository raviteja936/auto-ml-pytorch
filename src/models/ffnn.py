import sys
import torch.nn as nn
import torch.nn.functional as F
from src.utils.cli import CliArgs


class Net(nn.Module):
    def __init__(self, params, in_dim):
        super().__init__()
        self.params = params
        n_units = params.dense_layers
        out_dim = params.out_dim

        self.layers = []
        self.layers.append(nn.Linear(in_dim, n_units[0]))
        curr_dim = n_units[0]
        for i in range(1, len(n_units)):
            self.layers.append(nn.Linear(curr_dim, n_units[i]))

            curr_dim = n_units[i]
        self.final_layer = nn.Linear(curr_dim, out_dim)

    def forward(self, x):
        for layer in self.layers:
            # norm = nn.BatchNorm1d(layer.out_features)
            dropout = nn.Dropout(self.params.dropout_rate)
            x = layer(x)
            x = dropout(x)
            x = F.relu(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    args = CliArgs(sys.argv[1:])
    p = args.get_params()
    net = Net(p, 10)
    print(list(net.parameters()))
