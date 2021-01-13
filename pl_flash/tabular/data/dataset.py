import torch
from torch.utils.data import Dataset
import numpy as np


class PandasDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col, regression=False):
        cat_vars = [c.to_numpy().astype(np.int64) for n, c in df[cat_cols].items()]
        num_vars = [c.to_numpy().astype(np.float32) for n, c in df[num_cols].items()]

        self.target = df[target_col].to_numpy().astype(np.float32 if regression else np.int64)

        self.cat_vars = np.stack(cat_vars, 1)
        self.num_vars = np.stack(num_vars, 1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {"id": idx, "x": (self.cat_vars[idx], self.num_vars[idx]), "target": self.target[idx]}
