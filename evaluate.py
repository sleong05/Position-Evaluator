# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChessCNN().to(device)
model.load_state_dict(torch.load('chess_cnn_epoch3.pt', map_location=device))
model.eval()

test_dataset = ChessDataset('test.csv')
test_loader  = DataLoader(test_dataset, batch_size=512, num_workers=0, shuffle=False)

total_mae = 0
total_mse = 0
batches = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)

        total_mae += (pred - batch_y).abs().mean().item()
        total_mse += ((pred - batch_y) ** 2).mean().item()
        batches += 1

        if batches % 500 == 0:
            print(f"batch {batches} — mae: {total_mae/batches:.4f}")

print(f"\nFinal Test MAE: {total_mae/batches:.4f} pawns")
print(f"Final Test MSE: {total_mse/batches:.4f}")
print(f"Final Test RMSE: {(total_mse/batches)**0.5:.4f} pawns")
