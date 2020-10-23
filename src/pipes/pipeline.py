from src.pipes.dataset import CustomDataset
# from .preprocess import PackNumericFeatures


class DataPipe:
    def __init__(self, params, mode):
        self.params = params
        if mode == "train":
            self.is_training = 1

        self.features = {"target": self.params.layout["target"], "numeric": self.params.layout["numeric"],
                         "categorical": self.params.layout["categorical"]}

        self.train_file_path = self.params.train_path
        self.test_file_path = self.params.train_path
        # if not self.test_file_path:
            # self.train_file_path, \

    def build(self):
        train_data = CustomDataset(self.train_file_path, self.features)
        test_data = CustomDataset(self.test_file_path, self.features)
        return train_data, test_data

    def split_dataset(self, train_file_path):
        # TODO split train into train and test datasets
        pass
