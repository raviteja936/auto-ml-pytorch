import sys
import torch
from src.utils.cli import CliArgs
from src.pipes.pipeline import DataPipe
from src.models.ffnn import Net
from src.evaluate.evalloop import EvalLoop


def evaluate(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "evaluate")
    _, test_loader = pipe.build()

    net = Net(params, pipe.width)
    net.load_state_dict(torch.load(params.out_path))

    eval = EvalLoop(net, test_loader)
    test_accuracy = eval.predict()
    print("Validation Accuracy = %d%%" % (test_accuracy))


if __name__ == "__main__":
    evaluate(sys.argv[1:])
