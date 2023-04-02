import pickle
from typing import Union
import torch
import os
from torch_geometric.data import Dataset
from pathlib import Path


class GraphFileDataset(Dataset):
    def __init__(self, root: Path, transform=None, pre_transform=None, pre_filter=None, use_tqdm=False):
        if use_tqdm:
            from tqdm import tqdm
            self.tqdm = tqdm
        else:
            self.tqdm = lambda x: x

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        try:
            return [i for i in os.listdir(self.processed_dir) if i.startswith("data_")]
        except Exception:
            return ["1"]

    def process(self):
        idx = 0
        for raw_path in self.tqdm(self.raw_paths):
            data = torch.load(raw_path)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(
                self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    

    def from_pickle(filename: Union[str, Path]) -> 'GraphFileDataset':
        with open(filename, 'rb') as f:
            return pickle.load(f)
