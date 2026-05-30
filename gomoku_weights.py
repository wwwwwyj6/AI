"""Gomoku positional weight matrix for 15x15 board evaluation.

This module provides a static position weight matrix and helper functions
that use the same board-related names and constants as gomoku_pattern.py.
"""

from typing import List, Tuple, Iterable, Dict
from gomoku_pattern import sliding_five_windows

EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 15

_PATTERN_BONUS: Dict[str, int] = {
    "FOUR_LIVE": 170,
    "FOUR_SLEEP": 130,
    "FOUR_SPLIT": 100,
    "THREE_LIVE": 55,
    "THREE_GAP_LIVE": 48,
    "THREE_SLEEP": 42,
    "THREE_GAP_SLEEP": 34,
    "THREE_SPLIT": 24,
    "THREE_SPARSE": 15,
    "THREE_DEAD": 8,
    "TWO_OPEN": 12,
    "TWO_HALF_OPEN": 9,
    "TWO_GAP": 6,
    "TWO_SPLIT": 5,
    "TWO_SPARSE": 3,
    "ONE": 1,
    "EMPTY_FIVE": 0,
    "MIXED": 0,
}


def _pattern_bonus(pattern_name: str) -> int:
    for key, value in _PATTERN_BONUS.items():
        if pattern_name.endswith(key) or pattern_name == key:
            return value
    return 0


def point_pattern_weight(board: List[List[int]], y: int, x: int, player: int = None) -> int:
    """Return a per-point weight for an empty board location.

    Non-empty points return -1. Empty points accumulate the best pattern
    bonuses from each direction based on the given player's stones.
    If player is None, compute bonuses from any stone (original behavior).
    """
    if not (0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE):
        raise ValueError(f"Point out of range: {(y, x)}")

    if board[y][x] != EMPTY:
        return -1

    patterns = [
        w for w in sliding_five_windows(board)
        if (y, x) in w["coords"] and not w["pattern"].startswith("MIXED")
    ]
    if not patterns:
        return 0

    if player is not None:
        player_prefix = "BLACK" if player == BLACK else "WHITE"
        patterns = [w for w in patterns if w["pattern"].startswith(player_prefix)]
        if not patterns:
            return 0

    best_by_direction: Dict[Tuple[int, int], int] = {}
    count_by_direction: Dict[Tuple[int, int], int] = {}

    for w in patterns:
        direction = w["direction"]
        bonus = _pattern_bonus(w["pattern"])
        if bonus <= 0:
            continue
        current_best = best_by_direction.get(direction, 0)
        if bonus > current_best:
            best_by_direction[direction] = bonus
            count_by_direction[direction] = 1
        elif bonus == current_best:
            count_by_direction[direction] = count_by_direction.get(direction, 0) + 1

    total = 0
    for direction, best_bonus in best_by_direction.items():
        total += best_bonus * count_by_direction[direction]

    return total


def validate_board(board: List[List[int]]) -> None:
    if len(board) != BOARD_SIZE:
        raise ValueError(f"Board must be {BOARD_SIZE} rows")
    for row in board:
        if len(row) != BOARD_SIZE:
            raise ValueError(f"Board must be {BOARD_SIZE} columns")
        for cell in row:
            if cell not in (EMPTY, BLACK, WHITE):
                raise ValueError(f"Invalid cell value: {cell}")


def position_weight_matrix(board: List[List[int]]) -> List[List[int]]:
    """Return a 15x15 matrix of weights for empty board locations."""
    validate_board(board)
    return [
        [point_pattern_weight(board, y, x) for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]


def black_weight_matrix(board: List[List[int]]) -> List[List[int]]:
    """Return a 15x15 matrix of weights based on BLACK stone patterns."""
    validate_board(board)
    return [
        [point_pattern_weight(board, y, x, player=BLACK) for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]


def white_weight_matrix(board: List[List[int]]) -> List[List[int]]:
    """Return a 15x15 matrix of weights based on WHITE stone patterns."""
    validate_board(board)
    return [
        [point_pattern_weight(board, y, x, player=WHITE) for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]


def position_weight(board: List[List[int]], y: int, x: int) -> int:
    """Return the pattern-based weight for a single board coordinate."""
    return point_pattern_weight(board, y, x)


def evaluate_move(board: List[List[int]], y: int, x: int, player: int, mode: str = "balanced") -> int:
    """Evaluate the utility of placing a stone at (y, x) for the given player.

    mode can be:
    - 'attack': maximize self's pattern strength
    - 'defense': maximize opponent's threat, defend high-risk areas
    - 'balanced': weighted combination

    Returns a composite score based on player's patterns and opponent's threats.
    """
    if not (0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE):
        raise ValueError(f"Point out of range: {(y, x)}")
    if board[y][x] != EMPTY:
        return -1

    self_weight = point_pattern_weight(board, y, x, player=player)
    opponent = WHITE if player == BLACK else BLACK
    opponent_weight = point_pattern_weight(board, y, x, player=opponent)

    if mode == "attack":
        defense_bonus = int(opponent_weight * 0.6)
        return self_weight + defense_bonus
    elif mode == "defense":
        defense_bonus = int(opponent_weight * 1.2)
        return self_weight + defense_bonus
    else:
        mixed_weight = int(self_weight * 0.6 + opponent_weight * 0.4)
        return mixed_weight


def board_position_score(board: List[List[int]], player: int, use_pattern_weights: bool = True) -> int:
    """Compute a board score for the given player.

    If use_pattern_weights is True, this returns the sum of candidate empty-point
    weights for the board. Otherwise it returns the count of the player's stones.
    """
    validate_board(board)
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")
    if use_pattern_weights:
        return sum(
            point_pattern_weight(board, y, x)
            for y in range(BOARD_SIZE)
            for x in range(BOARD_SIZE)
            if board[y][x] == EMPTY
        )
    return sum(1 for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] == player)


def board_from_moves(moves: Iterable[Tuple[int, int, int]]) -> List[List[int]]:
    """Create a 15x15 board from an iterable of moves (player, y, x)."""
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for player, y, x in moves:
        if player not in (BLACK, WHITE):
            raise ValueError(f"Invalid player: {player}")
        if not (0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE):
            raise ValueError(f"Move out of range: {(y, x)}")
        if board[y][x] != EMPTY:
            raise ValueError(f"Cell already occupied: {(y, x)}")
        board[y][x] = player
    return board


if __name__ == "__main__":
    board = board_from_moves([(BLACK, 7, 7), (WHITE, 6, 7), (BLACK, 7, 8)])

    black_matrix = black_weight_matrix(board)
    white_matrix = white_weight_matrix(board)

    print("Black weight matrix (7,6):", black_matrix[7][6])
    print("White weight matrix (7,6):", white_matrix[7][6])
    print()
    print("Move evaluation (7,6) for BLACK:")
    print("  attack:", evaluate_move(board, 7, 6, BLACK, mode="attack"))
    print("  defense:", evaluate_move(board, 7, 6, BLACK, mode="defense"))
    print("  balanced:", evaluate_move(board, 7, 6, BLACK, mode="balanced"))
