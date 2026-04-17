# preprocess.py
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

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

print("Loading dataset...")
dataset = load_from_disk("chess_data")
train = dataset['train']
NUM_SHARDS = 13

total_saved = 0

for i in range(NUM_SHARDS):
    print(f"\nProcessing shard {i+1}/{NUM_SHARDS}...")
    shard = train.shard(num_shards=NUM_SHARDS, index=i)
    df = shard.to_pandas()
    print(f"  loaded {len(df)} rows")

    tqdm.pandas()
    df['evaluation'] = df['evaluation'].progress_apply(parse_eval)
    df = df.dropna(subset=['evaluation'])
    df = df[['fen', 'evaluation']]

    df.to_csv('chess_data.csv',
              mode='a',
              header=(i == 0),
              index=False)

    total_saved += len(df)
    print(f"  saved {len(df)} rows — running total: {total_saved}")

print(f"\nDone — {total_saved} total positions saved to chess_data.csv")