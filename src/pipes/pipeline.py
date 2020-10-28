from src.pipes.dataset import CustomDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
np.random.seed(0)


class DataPipe:
    def __init__(self, params, mode="train", preprocess=True):
        self.params = params
        if mode == "train":
            self.is_training = True
        else:
            self.is_training = False

        self.preprocess = preprocess
        self.features = {"target": self.params.layout["target"], "numeric": self.params.layout["numeric"],
                         "categorical": self.params.layout["categorical"]}
        self.num_kwargs = {}
        self.length = 0
        self.width = 0

    def build(self):
        cols = [self.params.layout["target"]] + self.params.layout["numeric"] + self.params.layout["categorical"]
        delimiter = None
        if hasattr(self.params, "delimiter"):
            delimiter = self.params.delimiter

        if self.is_training:
            train_df = pd.read_csv(self.params.train_path, usecols=cols, delimiter=delimiter)

            if not hasattr(self.params, "val_path"):
                train_df, val_df = self.split_dataset(train_df)
            else:
                val_df = pd.read_csv(self.params.val_path, usecols=cols, delimiter=delimiter)

            self.num_kwargs = self.get_num_kwargs(train_df[self.features["numeric"]])

            if self.preprocess:
                train_df = self.df_preprocess(train_df)
                train_df = pd.concat([train_df[self.features["target"]],
                                      self.num_preprocess(train_df[self.features["numeric"]]),
                                      self.cat_preprocess(train_df[self.features["categorical"]])], axis=1)

                val_df = self.df_preprocess(val_df)
                val_df = pd.concat([val_df[self.features["target"]],
                                    self.num_preprocess(val_df[self.features["numeric"]]),
                                    self.cat_preprocess(val_df[self.features["categorical"]])], axis=1)

            train_data = CustomDataset(train_df, self.features)
            val_data = CustomDataset(val_df, self.features)

            self.length = len(train_data)
            self.width = train_data[0]['x'].shape[0]

            train_loader = DataLoader(train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=0)

            return train_loader, val_loader

        else:
            test_df = pd.read_csv(self.params.test_path, usecols=cols, delimiter=delimiter)
            test_data = CustomDataset(test_df, self.features)
            self.length = len(test_data)
            self.width = test_data[0]['x'].shape[0]
            test_loader = DataLoader(test_data, batch_size=self.params.batch_size, shuffle=False, num_workers=0)
            return None, test_loader

    def split_dataset(self, df):
        msk = np.random.rand(len(df)) < 0.8
        train_df = df[msk]
        val_df = df[~msk]
        return train_df, val_df

    def get_num_kwargs(self, df):
        num_kwargs = {}
        num_kwargs["q1"] = df.quantile(0.01)
        num_kwargs["q99"] = df.quantile(0.99)
        num_kwargs["mean"] = df.mean()
        num_kwargs["std"] = df.std()
        return num_kwargs

    def df_preprocess(self, df):
        # df = df.dropna(axis=0, how="any")
        df = df.fillna(0)
        return df

    def num_preprocess(self, df_num):
        df_num = df_num.clip(lower=self.num_kwargs["q1"], upper=self.num_kwargs["q99"], axis=1)
        df_num = (df_num - self.num_kwargs["mean"]) / (0.001 + self.num_kwargs["std"])
        return df_num

    def cat_preprocess(self, df_cat):
        for col in df_cat.columns:
            df_cat = pd.concat([df_cat, pd.get_dummies(df_cat[col], prefix=col)], axis=1)
            del df_cat[col]
        return df_cat
