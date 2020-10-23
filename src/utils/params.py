import json


class Params:
    def __init__(self, path):
        with open(path, "r") as json_file:
            self.__dict__ = json.load(json_file)