# dataset_positional.py
from torch.utils.data import IterableDataset
import pandas as pd
import torch
import numpy as np
import chess
from fen_processing import board_to_tensor

def material_count(fen):
    board = chess.Board(fen)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    score = 0
    for piece, value in values.items():
        score += value * len(board.pieces(piece, chess.WHITE))
        score -= value * len(board.pieces(piece, chess.BLACK))
    return score

class ChessDatasetPositional(IterableDataset):
    def __init__(self, filepath, clip=20.0, chunksize=10000):
        self.filepath = filepath
        self.clip = clip
        self.chunksize = chunksize

    def __iter__(self):
        for chunk in pd.read_csv(self.filepath, chunksize=self.chunksize):
            chunk = chunk.sample(frac=1).reset_index(drop=True)
            fens = chunk['fen'].values
            evals = chunk['evaluation'].values
            for fen, val in zip(fens, evals):
                try:
                    x, turn = board_to_tensor(fen)
                    material = material_count(fen)
                    # flip material too if black to move
                    if turn == chess.BLACK:
                        material = -material
                    y = float(val)
                    if turn == chess.BLACK:
                        y = -y
                    # subtract material to get positional eval
                    y = np.float32(np.clip(y - material, -20.0, 20.0))
                    yield torch.tensor(x), torch.tensor(y)
                except:
                    continue