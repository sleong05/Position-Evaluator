import chess
import numpy as np
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def board_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    for i, piece in enumerate(piece_types):
        for sq in board.pieces(piece, chess.WHITE):
            tensor[i][sq // 8][sq % 8] = 1
        for sq in board.pieces(piece, chess.BLACK):
            tensor[i+6][sq // 8][sq % 8] = 1
    
    return tensor

tensor = board_to_tensor(fen)
print(tensor)