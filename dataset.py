# dataset.py
from torch.utils.data import IterableDataset
import pandas as pd
import torch
import numpy as np
import chess
from fen_processing import board_to_tensor

class ChessDataset(IterableDataset):
    def __init__(self, filepath, clip=20.0, chunksize=10000):
        self.filepath = filepath
        self.clip = clip
        self.chunksize = chunksize

    def __iter__(self):
        for chunk in pd.read_csv(self.filepath, chunksize=self.chunksize):
            chunk = chunk.sample(frac=1).reset_index(drop=True)  # shuffle chunk
            fens = chunk['fen'].values
            evals = chunk['evaluation'].values
            for fen, val in zip(fens, evals):
                try:
                    x, turn = board_to_tensor(fen)
                    y = np.float32(np.clip(val, -20.0, 20.0))
                    if turn == chess.BLACK:
                        y = -y  # flip eval for black to move
                    yield torch.tensor(x), torch.tensor(y)
                except:
                    continue