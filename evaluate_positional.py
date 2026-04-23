# evaluate_positional.py
import torch
import chess
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import ChessCNN
from fen_processing import board_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChessCNN().to(device)
model.load_state_dict(torch.load('chess_cnn_positional_best.pt', map_location=device))
model.eval()

def material_count(fen):
    board = chess.Board(fen)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    score = 0
    for piece, value in values.items():
        score += value * len(board.pieces(piece, chess.WHITE))
        score -= value * len(board.pieces(piece, chess.BLACK))
    return score

total_mae = 0
batches = 0

print("Evaluating positional model...")
for chunk in pd.read_csv('test.csv', chunksize=10000):
    for fen, val in zip(chunk['fen'].values, chunk['evaluation'].values):
        try:
            x, turn = board_to_tensor(fen)
            x = torch.tensor(x).unsqueeze(0).to(device)
            
            material = material_count(fen)
            actual = np.clip(float(val), -20.0, 20.0)
            
            with torch.no_grad():
                pred_positional = model(x).item()
            
            # add material back for fair comparison
            if turn == chess.BLACK:
                pred_full = -pred_positional + material
            else:
                pred_full = pred_positional + material
            
            total_mae += abs(pred_full - actual)
            batches += 1
        except:
            continue

    if batches % 100000 == 0:
        print(f"  {batches:,} positions — MAE so far: {total_mae/batches:.4f}")

print(f"\nPositional Model Full Eval MAE: {total_mae/batches:.4f} pawns")