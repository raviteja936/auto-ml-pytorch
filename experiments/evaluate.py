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

    net = torch.load(params.out_path)
    net.eval()

    eval = EvalLoop(net, test_loader)
    accuracy = eval.predict()
    print("Validation Accuracy = %d%%" % accuracy)

def score(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, mode="evaluate", preprocess=False)
    _, test_loader = pipe.build()

    net = torch.load(params.out_path)
    net.eval()

    eval = EvalLoop(net, test_loader)
    out_scores = eval.score()
    return out_scores

if __name__ == "__main__":
    evaluate(sys.argv[1:])
