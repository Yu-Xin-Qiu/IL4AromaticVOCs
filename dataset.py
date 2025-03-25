from __future__ import print_function, division
import functools
import  numpy  as  np
import torch
from torch.utils.data import Dataset




from tqdm import tqdm
import numpy as np
import torch
import functools
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['Normalized SMILES']

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens = np.array(
            list(Parallel(n_jobs=10)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles, desc='Tokenizing SMILES', total=len(self.smiles))
            ))
        )
        self.label = df[target]
        self.tokenizer = tokenizer
        self.t = df['T/K']


    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        smiles = self.smiles[index]
        t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
        return (X, t), y.float(), smiles, t

