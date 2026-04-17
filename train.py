# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessCNN
from datasets import load_from_disk

# ---- Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = ChessCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.HuberLoss()

# ---- Dataloaders ----
from datasets import load_from_disk

hf_dataset = load_from_disk("chess_data")

# split 90/10
split = hf_dataset.train_test_split(test_size=0.1)
train_dataset = ChessDataset(split['train'])
test_dataset  = ChessDataset(split['test'])

train_loader = DataLoader(train_dataset, batch_size=512, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=512, num_workers=4)

# ---- Training Loop ----
def train_epoch():
    model.train()
    total_loss, total_mae, batches = 0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae  += (pred - batch_y).abs().mean().item()
        batches    += 1

    return total_loss / batches, total_mae / batches

def test_epoch():
    model.eval()
    total_loss, total_mae, batches = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            total_loss += loss.item()
            total_mae  += (pred - batch_y).abs().mean().item()
            batches    += 1

    return total_loss / batches, total_mae / batches

# ---- Run ----
EPOCHS = 3

for epoch in range(EPOCHS):
    train_loss, train_mae = train_epoch()
    test_loss,  test_mae  = test_epoch()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f}")

# ---- Save ----
torch.save(model.state_dict(), 'chess_cnn.pt')
print("Model saved to chess_cnn.pt")