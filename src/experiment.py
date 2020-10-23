import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.cli import CliArgs
from src.pipes.pipeline import DataPipe
from src.models.ffnn import Net
from src.train.trainloop import TrainLoop


def experiment(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "train")
    train_data, val_data = pipe.build()
    net = Net(params, train_data[1]['x'].shape[0])
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=params.batch_size, shuffle=False, num_workers=0)

    loss_fn = getattr(nn, params.loss)()
    optimizer = getattr(optim, params.optimizer)(net.parameters(), lr=params.learning_rate, momentum=params.momentum)

    print_every = int(len(train_data) / (10 * params.batch_size))
    train = TrainLoop(net, train_loader, optimizer, loss_fn, val_loader=val_loader,
                      print_every=print_every)
    train.fit(params.num_epochs)


if __name__ == "__main__":
    experiment(sys.argv[1:])
