from __future__ import print_function, division
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, df, features, transform=None):
        self.df = df
        self.features = features
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = self.features['target']
        columns = self.features['numeric'] + self.features['categorical']
        x = torch.tensor(self.df[columns].iloc[idx]).type(torch.float)
        y = torch.tensor(self.df[target].iloc[idx]).type(torch.long)
        sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample
