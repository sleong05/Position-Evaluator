import csv
import chess



INPUT_FILE = "train.csv"          # must contain a column named 'fen'
OUTPUT_FILE = "fen_phase.csv"

PIECE_VALUES = {
    chess.QUEEN: 4,
    chess.ROOK: 2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}

def classify_phase(fen: str) -> str:
    board = chess.Board(fen)

    total_non_pawn_material = 0
    for piece_type, value in PIECE_VALUES.items():
        total_non_pawn_material += len(board.pieces(piece_type, chess.WHITE)) * value
        total_non_pawn_material += len(board.pieces(piece_type, chess.BLACK)) * value

    if total_non_pawn_material >= 40:
        return "early"
    elif total_non_pawn_material >= 15:
        return "middle"
    else:
        return "end"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["fen", "phase"])
    writer.writeheader()

    for i, row in enumerate(reader):

        fen = row["fen"].strip()
        phase = classify_phase(fen)

        print(f"{fen} -> {phase}")  # 👈 prints output

        writer.writerow({"fen": fen, "phase": phase})

print(f"Created {OUTPUT_FILE}")