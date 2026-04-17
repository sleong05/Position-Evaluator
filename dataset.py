from torch.utils.data import IterableDataset
import pandas as pd
import torch
import numpy as np
from fen_processing import board_to_tensor
from datasets import load_from_disk

dataset = load_from_disk("chess_data")

class ChessDataset(IterableDataset):
    def __init__(self, hf_dataset, clip=20.0):
        self.data = hf_dataset
        self.clip = clip

    def __iter__(self):
        for row in self.data:
            try:
                x = board_to_tensor(row['fen'])
                y = np.float32(np.clip(row['evaluation'], -self.clip, self.clip))
                yield torch.tensor(x), torch.tensor(y)
            except:
                continue
