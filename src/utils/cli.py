import argparse
import os
from .params import Params


class CliArgs:
    def __init__(self, args):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-p",
            "--params_path",
            help="path to params.json file path that contains input parameters"
        )

        self.parser.add_argument(
            "-o",
            "--out_path",
            help="path to directory to store output from the run",
            action="store_true"
        )

        self.args = self.parser.parse_args(args)

    def get_params(self):
        assert os.path.isfile(self.args.params_path), "No json configuration file found at {}".format(self.args.params_path)
        params = Params(self.args.params_path)
        return params

    def get_out_path(self):
        out_path = self.args.out_path
        if out_path is None:
            if os.path.isdir(os.path.join(self.args.params_path, "experiments")):
                return os.path.join(self.args.params_path, "experiments", "run_%s") % len(next(os.walk('examples'))[1])
            return os.path.join(self.args.params_path, "experiments", "run_1")

        return out_path
