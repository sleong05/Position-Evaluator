# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessCNN

# ---- Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = ChessCNN().to(device)
model.load_state_dict(torch.load('chess_cnn_epoch3.pt'))  # continue from epoch 3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lower lr
criterion = nn.HuberLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

# ---- Dataloaders ----
train_dataset = ChessDataset('train.csv')
test_dataset  = ChessDataset('test.csv')

train_loader = DataLoader(train_dataset, batch_size=512, num_workers=0, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=512, num_workers=0, shuffle=False)

# ---- Training Loop ----
def train_epoch(epoch):
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
        if batches % 1000 == 0:
            print(f"  epoch {epoch} batch {batches} — loss: {total_loss/batches:.4f} mae: {total_mae/batches:.4f}")
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

# ---- Early Stopping ----
best_mae = float('inf')
patience_count = 0
PATIENCE = 3

for epoch in range(1, 20):
    print(f"\nEpoch {epoch}")
    train_loss, train_mae = train_epoch(epoch)
    test_loss,  test_mae  = test_epoch()
    print(f"Epoch {epoch} done | Train MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    scheduler.step(test_mae)

    if test_mae < best_mae:
        best_mae = test_mae
        patience_count = 0
        torch.save(model.state_dict(), 'chess_cnn_best.pt')
        print(f"New best model saved — MAE: {best_mae:.4f}")
    else:
        patience_count += 1
        print(f"No improvement — patience {patience_count}/{PATIENCE}")
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nTraining complete — best Test MAE: {best_mae:.4f}")