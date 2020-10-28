import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.cli import CliArgs
from src.pipes.pipeline import DataPipe
from src.models.ffnn import Net
from src.train.trainloop import TrainLoop


def experiment(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "train")
    train_loader, val_loader = pipe.build()

    net = Net(params, pipe.width)
    loss_wt = None
    if hasattr(params, "loss_weights"):
        loss_wt = params.loss_weights
    loss_fn = getattr(nn, params.loss)(torch.FloatTensor(loss_wt))
    optimizer = getattr(optim, params.optimizer)(net.parameters(), lr=params.learning_rate, momentum=params.momentum)
    print_every = int(pipe.length / (2 * params.batch_size))

    train = TrainLoop(net, train_loader, optimizer, loss_fn, val_loader=val_loader,
                      print_every=print_every)
    train.fit(params.num_epochs)

    if hasattr(params, "out_path"):
        torch.save(net.state_dict(), params.out_path)


if __name__ == "__main__":
    experiment(sys.argv[1:])
