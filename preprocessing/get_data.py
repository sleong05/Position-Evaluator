# prepare_data.py
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import pandas as pd
import os

def parse_eval(val):
    val = str(val).strip()
    if val.startswith('M'):
        number = val[1:]
        if number.startswith('-'):
            return -20.0
        else:
            return 20.0
    try:
        return float(val)
    except:
        return None

# ---- Download ----
print("Downloading dataset...")
dataset = load_dataset("bingbangboom/stockfish-evaluation-SAN")
dataset.save_to_disk("chess_data")
print("Download done")

# ---- Preprocess ----
print("\nPreprocessing...")
train = load_from_disk("chess_data")['train']
NUM_SHARDS = 13
total_saved = 0

for i in range(NUM_SHARDS):
    print(f"  shard {i+1}/{NUM_SHARDS}...")
    shard = train.shard(num_shards=NUM_SHARDS, index=i)
    df = shard.to_pandas()
    tqdm.pandas()
    df['evaluation'] = df['evaluation'].progress_apply(parse_eval)
    df = df.dropna(subset=['evaluation'])
    df = df[['fen', 'evaluation']]
    df.to_csv('chess_data.csv', mode='a', header=(i == 0), index=False)
    total_saved += len(df)
    print(f"  running total: {total_saved:,}")

print(f"Preprocessing done — {total_saved:,} positions")

# ---- Split ----
print("\nSplitting into train and test...")
train_file = open('train.csv', 'w')
test_file  = open('test.csv', 'w')
header_written = False
total_train = 0
total_test  = 0

for i, chunk in enumerate(pd.read_csv('chess_data.csv', chunksize=10000)):
    if not header_written:
        chunk.iloc[:0].to_csv(train_file, index=False)
        chunk.iloc[:0].to_csv(test_file, index=False)
        header_written = True
    if i % 10 == 0:
        chunk.to_csv(test_file, index=False, header=False)
        total_test += len(chunk)
    else:
        chunk.to_csv(train_file, index=False, header=False)
        total_train += len(chunk)
    if i % 100 == 0:
        print(f"  processed {i * 10000:,} rows...")

train_file.close()
test_file.close()

# ---- Cleanup ----
print("\nCleaning up...")
os.remove('chess_data.csv')
import shutil
shutil.rmtree('chess_data')

print(f"\nAll done!")
print(f"train.csv — {total_train:,} rows")
print(f"test.csv  — {total_test:,} rows")