"""
Interactive minimax chess engine using the trained CNN evaluation.

Usage:
    python play_cnn.py

You'll be prompted for moves in UCI format (e.g., 'e2e4', 'g1f3', 'e7e8q' for promotion).
The engine will respond with its chosen move, which you then play on chess.com.
Type 'quit' to exit, 'undo' to take back the last full move pair, 'fen' to print FEN,
'board' to redisplay, 'depth N' to change search depth.
"""

import argparse
import sys
import chess
import chess.pgn
import torch
import torch.nn as nn
import numpy as np


# ---------- Model (must match training architecture) ----------

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.convs(x)).squeeze(1)


# ---------- Board encoding (v2: 6-plane side-to-move normalized) ----------

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode board matching training's fen_processing.py exactly.

    - 6 planes, one per piece type, in order [P, N, B, R, Q, K]
    - +1 for side-to-move pieces, -1 for opponent pieces
    - Row index = sq // 8 (no vertical flip), col = sq % 8
    """
    tensor = np.zeros((6, 8, 8), dtype=np.float32)
    us = board.turn
    them = not board.turn
    for i, piece in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece, us):
            tensor[i][sq // 8][sq % 8] = 1.0
        for sq in board.pieces(piece, them):
            tensor[i][sq // 8][sq % 8] = -1.0
    return tensor


# ---------- Evaluation ----------

@torch.no_grad()
def evaluate_positions(model, boards, device):
    """Batch-evaluate a list of boards. Returns evals in pawns from side-to-move perspective."""
    if not boards:
        return np.array([])
    tensors = np.stack([board_to_tensor(b) for b in boards])
    x = torch.from_numpy(tensors).to(device)
    evals = model(x).cpu().numpy()
    return evals


def terminal_eval(board: chess.Board) -> float:
    """Return evaluation for terminal positions from side-to-move perspective."""
    if board.is_checkmate():
        return -20.0  # side to move is checkmated
    return 0.0  # draw


# ---------- Minimax with alpha-beta ----------

def negamax(model, board, depth, alpha, beta, device):
    """Negamax: always returns score from side-to-move perspective."""
    if board.is_game_over():
        return terminal_eval(board)

    if depth == 0:
        evals = evaluate_positions(model, [board], device)
        return float(evals[0])

    legal_moves = list(board.legal_moves)

    # Move ordering: evaluate resulting positions in batch, sort descending
    # (we want moves that lead to bad positions for opponent = good for us)
    if depth >= 2 and len(legal_moves) > 1:
        child_boards = []
        for m in legal_moves:
            board.push(m)
            child_boards.append(board.copy(stack=False))
            board.pop()
        child_evals = evaluate_positions(model, child_boards, device)
        # Child eval is from opponent's perspective; lower is better for us
        order = np.argsort(child_evals)
        legal_moves = [legal_moves[i] for i in order]

    best = -float('inf')
    for move in legal_moves:
        board.push(move)
        score = -negamax(model, board, depth - 1, -beta, -alpha, device)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_move(model, board, depth, device):
    """Pick the best move for the side to move. Returns (move, score)."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    # Order root moves by 1-ply eval
    child_boards = []
    for m in legal_moves:
        board.push(m)
        child_boards.append(board.copy(stack=False))
        board.pop()
    child_evals = evaluate_positions(model, child_boards, device)
    order = np.argsort(child_evals)  # ascending: best for us first
    legal_moves = [legal_moves[i] for i in order]

    best_move = legal_moves[0]
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for move in legal_moves:
        board.push(move)
        score = -negamax(model, board, depth - 1, -beta, -alpha, device)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
    return best_move, best_score


# ---------- Interactive loop ----------

def print_board(board):
    print()
    print(board.unicode(borders=True, empty_square='.'))
    print(f"  {'White' if board.turn == chess.WHITE else 'Black'} to move  "
          f"(move {board.fullmove_number}, ply {board.ply()})")
    if board.is_check():
        print("  CHECK")
    print()


def parse_move(board, s):
    """Parse a move in UCI or SAN. Returns chess.Move or None."""
    s = s.strip()
    # Try UCI first
    try:
        m = chess.Move.from_uci(s)
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    # Try SAN
    try:
        return board.parse_san(s)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_cnn_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--depth', type=int, default=3,
                        help='Minimax search depth (default 3)')
    parser.add_argument('--engine-color', choices=['white', 'black', 'ask'],
                        default='ask', help='Which color the engine plays')
    parser.add_argument('--fen', default=None,
                        help='Starting FEN (default: standard starting position)')
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} on device {device}...")

    model = ChessCNN().to(device)
    state = torch.load(args.model, map_location=device)
    # Handle checkpoints saved as {'model_state_dict': ...} or raw state dicts
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")

    board = chess.Board(args.fen) if args.fen else chess.Board()

    # Decide engine color
    if args.engine_color == 'ask':
        while True:
            ans = input("Does the engine play (w)hite or (b)lack? ").strip().lower()
            if ans in ('w', 'white'):
                engine_color = chess.WHITE
                break
            if ans in ('b', 'black'):
                engine_color = chess.BLACK
                break
    else:
        engine_color = chess.WHITE if args.engine_color == 'white' else chess.BLACK

    depth = args.depth
    print(f"Engine plays {'White' if engine_color == chess.WHITE else 'Black'}, "
          f"search depth = {depth}")
    print("Commands: 'quit', 'undo', 'fen', 'board', 'depth N', or enter a move (UCI or SAN).\n")
    print_board(board)

    while not board.is_game_over():
        if board.turn == engine_color:
            print("Engine thinking...")
            move, score = choose_move(model, board, depth, device)
            if move is None:
                break
            san = board.san(move)
            board.push(move)
            print(f"Engine plays: {san}  ({move.uci()})  eval={score:+.2f} pawns")
            print_board(board)
        else:
            cmd = input("Your move (opponent's move on chess.com): ").strip()
            if not cmd:
                continue
            low = cmd.lower()
            if low in ('quit', 'exit', 'q'):
                break
            if low == 'board':
                print_board(board)
                continue
            if low == 'fen':
                print(board.fen())
                continue
            if low == 'undo':
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("Took back last full move.")
                elif len(board.move_stack) == 1:
                    board.pop()
                    print("Took back last half-move.")
                else:
                    print("Nothing to undo.")
                print_board(board)
                continue
            if low.startswith('depth'):
                parts = low.split()
                if len(parts) == 2 and parts[1].isdigit():
                    depth = int(parts[1])
                    print(f"Search depth set to {depth}.")
                else:
                    print("Usage: depth N")
                continue
            move = parse_move(board, cmd)
            if move is None:
                print(f"Invalid/illegal move: '{cmd}'. Try UCI (e2e4) or SAN (Nf3).")
                continue
            board.push(move)
            print_board(board)

    # Game over
    print("=" * 40)
    print("Game over.")
    print(f"Result: {board.result()}")
    if board.is_checkmate():
        winner = 'Black' if board.turn == chess.WHITE else 'White'
        print(f"Checkmate. {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate.")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.can_claim_fifty_moves():
        print("Draw by fifty-move rule.")
    elif board.can_claim_threefold_repetition():
        print("Draw by threefold repetition.")
    print()
    print("PGN:")
    # Build a minimal PGN
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "CNN vs chess.com bot"
    game.headers["White"] = "CNN" if engine_color == chess.WHITE else "Opponent"
    game.headers["Black"] = "CNN" if engine_color == chess.BLACK else "Opponent"
    game.headers["Result"] = board.result()
    print(game)


if __name__ == '__main__':
    main()