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
        # norm = nn.BatchNorm1d(layer.out_features)
        self.layers.append((nn.Linear(in_dim, n_units[0]), nn.BatchNorm1d(n_units[0])))
        curr_dim = n_units[0]
        for i in range(1, len(n_units)):
            self.layers.append((nn.Linear(curr_dim, n_units[i]), nn.BatchNorm1d(n_units[i])))

            curr_dim = n_units[i]
        self.final_layer = nn.Linear(curr_dim, out_dim)
        self.drop_layer = nn.Dropout(self.params.dropout_rate)

    def forward(self, x):
        for layer in self.layers:

            x = layer[0](x)
            # x = layer[1](x)
            x = self.drop_layer(x)
            x = F.relu(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    args = CliArgs(sys.argv[1:])
    p = args.get_params()
    net = Net(p, 10)
    print(list(net.parameters()))
