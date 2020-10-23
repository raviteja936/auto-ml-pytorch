from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, csv_file, features, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features = features
        cols = [features['target']] + features['numeric'] + features['categorical']
        self.df = pd.read_csv(csv_file, usecols=cols)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.df.loc[idx, self.df.columns != self.features['target']].values).type(torch.float)
        y = torch.tensor(self.df.loc[idx, self.df.columns == self.features['target']].values).type(torch.float)
        sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    f = {"target": "bandgap_energy_ev", "numeric": ["lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang"],
                         "categorical": ["spacegroup"]}
    dataset = CustomDataset("./data/train.csv", f)
    sample = dataset[1, 2, 3]
    print(sample['x'].shape, sample['y'].shape)
    print(len(dataset))
    print(dataset)
