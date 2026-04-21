# baseline.py
import chess
import pandas as pd
import numpy as np
from tqdm import tqdm

def material_eval(fen):
    board = chess.Board(fen)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    score = 0
    for piece, value in values.items():
        score += value * len(board.pieces(piece, chess.WHITE))
        score -= value * len(board.pieces(piece, chess.BLACK))
    return score

print("Loading test data...")
total_mae = 0
total_mse = 0
count = 0

for chunk in tqdm(pd.read_csv('test.csv', chunksize=10000)):
    for fen, val in zip(chunk['fen'].values, chunk['evaluation'].values):
        try:
            pred = material_eval(fen)
            actual = np.clip(float(val), -20.0, 20.0)
            error = abs(pred - actual)
            total_mae += error
            total_mse += error ** 2
            count += 1
        except:
            continue

mae  = total_mae / count
rmse = (total_mse / count) ** 0.5

print(f"\nBaseline Results:")
print(f"Positions evaluated: {count:,}")
print(f"MAE:  {mae:.4f} pawns")
print(f"RMSE: {rmse:.4f} pawns")
print(f"\nFor comparison:")
print(f"CNN Test MAE: ~2.18 pawns")
print(f"Baseline MAE: {mae:.4f} pawns")