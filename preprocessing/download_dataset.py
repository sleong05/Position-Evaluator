from datasets import load_dataset

# download and save to disk — run this once
dataset = load_dataset("bingbangboom/stockfish-evaluation-SAN")
dataset.save_to_disk("chess_data")
print("done")