from __future__ import print_function, division
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, df, features, transform=None):
        self.df = df
        self.features = features
        self.transform = transform
        # print(df.shape)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.df_target = self.df[self.features['target']]
        x = torch.tensor(self.df.iloc[idx]).type(torch.float)
        y = torch.tensor(self.df_target.iloc[idx]).type(torch.long)
        sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample
