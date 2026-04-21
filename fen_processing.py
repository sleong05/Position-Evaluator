# fen_processing.py
import chess
import numpy as np

def board_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((6, 8, 8), dtype=np.float32)
    
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    us   = board.turn
    them = not board.turn
    
    for i, piece in enumerate(piece_types):
        for sq in board.pieces(piece, us):
            tensor[i][sq // 8][sq % 8] = 1.0
        for sq in board.pieces(piece, them):
            tensor[i][sq // 8][sq % 8] = -1.0
    
    return tensor, board.turn  # return turn so dataset can flip eval