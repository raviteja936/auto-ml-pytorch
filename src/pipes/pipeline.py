from src.pipes.dataset import CustomDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
# from .preprocess import PackNumericFeatures


class DataPipe:
    def __init__(self, params, mode="train"):
        self.params = params
        if mode == "train":
            self.is_training = True
        self.features = {"target": self.params.layout["target"], "numeric": self.params.layout["numeric"],
                         "categorical": self.params.layout["categorical"]}
        self.length = 0
        self.width = 0

    def build(self):
        cols = [self.params.layout["target"]] + self.params.layout["numeric"] + self.params.layout["categorical"]
        delimiter = None
        if hasattr(self.params, "delimiter"):
            delimiter = self.params.delimiter
        if self.is_training:
            train_df = pd.read_csv(self.params.train_path, usecols=cols, delimiter=delimiter)

            if hasattr(self.params, "val_path"):
                val_df = pd.read_csv(self.params.val_path, usecols=cols, delimiter=delimiter)
            else:
                train_df, val_df = self.split_dataset(train_df)

            train_data = CustomDataset(train_df, self.features)
            val_data = CustomDataset(val_df, self.features)

            self.length = len(train_data)
            self.width = train_data[0]['x'].shape[0]
            # print(self.length, self.width)

            train_loader = DataLoader(train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=0)

            return train_loader, val_loader

        else:
            val_df = pd.read_csv(self.params.val_path, usecols=cols, delimiter=delimiter)
            val_data = CustomDataset(val_df, self.features)
            val_loader = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=0)
            return None, val_loader

    def split_dataset(self, df):
        msk = np.random.rand(len(df)) < 0.7
        train_df = df[msk]
        val_df = df[~msk]
        return train_df, val_df
