
#evaluator
"""Gomoku board evaluator used by search leaf nodes.

Provides `evaluate(board, player, ...)` which returns a signed score:
positive = `player` is better, negative = opponent is better.

This module reuses names and functions from gomoku_weights.py and gomoku_pattern.py
so variable names are consistent with the rest of the project.
"""
from typing import List

from gomoku_pattern import BLACK, WHITE, EMPTY, BOARD_SIZE, first_five_winner
from gomoku_weights import (
    validate_board,
    board_position_score,
    board_pattern_threat_score,
    board_from_moves,
)


def evaluate(
    board: List[List[int]],
    player: int,
    use_position_weight: bool = True,
    use_pattern_weight: bool = True,
    attack_weight: float = 1.0,
    defense_weight: float = 0.75,
) -> int:
    """Evaluate the given 15x15 `board` for `player`.

    Args:
        board: 15x15 matrix with values EMPTY/BLACK/WHITE.
        player: BLACK or WHITE - the side for which the score is positive.
        use_position_weight: if True, use positional evaluation instead of stone count.
        use_pattern_weight: if True, use pattern-derived position weights;
            if False, use simple static stone-position weights.
        attack_weight: multiplier applied to the player's threat score.
        defense_weight: multiplier applied to the opponent's threat score.

    Returns:
        Signed integer score: positive means `player` advantage; negative means opponent advantage.
        The magnitude reflects the relative board strength, so larger absolute values
        correspond to a more decisive advantage.
    """
    validate_board(board)
    if player not in (BLACK, WHITE):
        raise ValueError("player must be BLACK or WHITE")

    opponent = WHITE if player == BLACK else BLACK

    winner = first_five_winner(board)
    if winner == player:
        return 100000
    if winner == opponent:
        return -100000

    player_score = board_position_score(
        board,
        player,
        use_position_weight=use_position_weight,
        use_pattern_weight=use_pattern_weight,
    )
    opponent_score = board_position_score(
        board,
        opponent,
        use_position_weight=use_position_weight,
        use_pattern_weight=use_pattern_weight,
    )

    player_threat = board_pattern_threat_score(board, player)
    opponent_threat = board_pattern_threat_score(board, opponent)
    base_score = player_score - opponent_score

    return int(base_score + player_threat * attack_weight - opponent_threat * defense_weight)


# convenience alias used by search routines at leaf nodes
def evaluate_leaf(board: List[List[int]], player: int, **kwargs) -> int:
    """Alias for evaluate(), intended for leaf-node evaluation in search."""
    return evaluate(board, player, **kwargs)


if __name__ == "__main__":
    # simple demo
    from gomoku_pattern import board_from_moves, BLACK, WHITE

    board = board_from_moves([
        (BLACK, 7, 7),
        (BLACK, 7, 8),
        (BLACK, 7, 9),
        (BLACK, 7, 10),
        (WHITE, 6, 7),
    ])

    print("evaluate (pattern weights) for BLACK:", evaluate(board, BLACK, use_position_weight=True, use_pattern_weight=True))
    print("evaluate (pattern weights) for WHITE:", evaluate(board, WHITE, use_position_weight=True, use_pattern_weight=True))
    print("evaluate (static position weights) for BLACK:", evaluate(board, BLACK, use_position_weight=True, use_pattern_weight=False))
    print("evaluate (stone-count only) for WHITE:", evaluate(board, WHITE, use_position_weight=False))




#patterns
"""Gomoku pattern recognition using sliding 5-cell windows.

This module scans a 15x15 board for all contiguous 5-cell segments in
horizontal, vertical, and diagonal directions. It supports identifying
patterns by counts and returns the exact cells and values for each quintuple.

Board values:
- 0 = empty
- 1 = black
- 2 = white
"""

from typing import List, Tuple, Iterable, Dict

EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 15
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

PatternInfo = Dict[str, object]


def validate_board(board: List[List[int]]) -> None:
    if len(board) != BOARD_SIZE:
        raise ValueError(f"Board must be {BOARD_SIZE} rows")
    for row in board:
        if len(row) != BOARD_SIZE:
            raise ValueError(f"Board must be {BOARD_SIZE} columns")
        for cell in row:
            if cell not in (EMPTY, BLACK, WHITE):
                raise ValueError(f"Invalid cell value: {cell}")


def _segments(values: Tuple[int, int, int, int, int], player: int) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(values):
        if value == player:
            if start is None:
                start = index
        elif start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _outer_value(board: List[List[int]], coord: Tuple[int, int], dy: int, dx: int) -> int:
    y, x = coord[0] + dy, coord[1] + dx
    if 0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE:
        return board[y][x]
    return -1


def _segment_openness(board: List[List[int]], coords: Tuple[Tuple[int, int], ...], segment: Tuple[int, int], direction: Tuple[int, int]) -> str:
    dy, dx = direction
    left = _outer_value(board, coords[segment[0]], -dy, -dx)
    right = _outer_value(board, coords[segment[1]], dy, dx)
    left_open = left == EMPTY
    right_open = right == EMPTY
    if left_open and right_open:
        return "OPEN"
    if left_open or right_open:
        return "HALF_OPEN"
    return "CLOSED"


def _run_lengths(values: Tuple[int, int, int, int, int], player: int) -> List[int]:
    return [end - start + 1 for start, end in _segments(values, player)]


def _pattern_name(values: Tuple[int, int, int, int, int], board: List[List[int]], coords: Tuple[Tuple[int, int], ...], direction: Tuple[int, int]) -> str:
    black = values.count(BLACK)
    white = values.count(WHITE)
    if black > 0 and white > 0:
        return f"MIXED_B{black}_W{white}"

    if black == 0 and white == 0:
        return "EMPTY_FIVE"

    player = BLACK if black > 0 else WHITE
    prefix = "BLACK" if player == BLACK else "WHITE"
    count = black or white
    segments = _segments(values, player)

    if count == 5:
        return f"{prefix}_FIVE"

    if count == 4:
        if len(segments) == 1:
            openness = _segment_openness(board, coords, segments[0], direction)
            if openness == "OPEN":
                return f"{prefix}_FOUR_LIVE"
            return f"{prefix}_FOUR_SLEEP"
        return f"{prefix}_FOUR_SPLIT"

    if count == 3:
        if len(segments) == 1 and segments[0][1] - segments[0][0] + 1 == 3:
            openness = _segment_openness(board, coords, segments[0], direction)
            if openness == "OPEN":
                return f"{prefix}_THREE_LIVE"
            if openness == "HALF_OPEN":
                return f"{prefix}_THREE_SLEEP"
            return f"{prefix}_THREE_DEAD"
        if len(segments) == 2:
            gap = segments[1][0] - segments[0][1] - 1
            if gap == 1:
                first_open = _segment_openness(board, coords, segments[0], direction)
                second_open = _segment_openness(board, coords, segments[1], direction)
                if first_open == "OPEN" and second_open == "OPEN":
                    return f"{prefix}_THREE_GAP_LIVE"
                return f"{prefix}_THREE_GAP_SLEEP"
            return f"{prefix}_THREE_SPLIT"
        return f"{prefix}_THREE_SPARSE"

    if count == 2:
        if len(segments) == 1:
            openness = _segment_openness(board, coords, segments[0], direction)
            return f"{prefix}_TWO_{openness}"
        if len(segments) == 2:
            gap = segments[1][0] - segments[0][1] - 1
            if gap == 1:
                return f"{prefix}_TWO_GAP"
            return f"{prefix}_TWO_SPLIT"
        return f"{prefix}_TWO_SPARSE"

    if count == 1:
        return f"{prefix}_ONE"

    return "UNKNOWN"


def sliding_five_windows(board: List[List[int]]) -> List[PatternInfo]:
    """Return every 5-cell window on the board with direction and pattern info."""
    validate_board(board)
    windows: List[PatternInfo] = []

    for dy, dx in _DIRECTIONS:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                end_y = y + dy * 4
                end_x = x + dx * 4
                if not (0 <= end_y < BOARD_SIZE and 0 <= end_x < BOARD_SIZE):
                    continue

                coords: List[Tuple[int, int]] = []
                values: List[int] = []
                for step in range(5):
                    py = y + dy * step
                    px = x + dx * step
                    coords.append((py, px))
                    values.append(board[py][px])

                pattern_name = _pattern_name(tuple(values), board, tuple(coords), (dy, dx))
                windows.append(
                    {
                        "coords": tuple(coords),
                        "values": tuple(values),
                        "direction": (dy, dx),
                        "pattern": pattern_name,
                        "black": values.count(BLACK),
                        "white": values.count(WHITE),
                        "empty": values.count(EMPTY),
                        "mixed": values.count(BLACK) > 0 and values.count(WHITE) > 0,
                        "black_runs": _run_lengths(tuple(values), BLACK),
                        "white_runs": _run_lengths(tuple(values), WHITE),
                    }
                )

    return windows


def find_winning_lines(board: List[List[int]]) -> List[PatternInfo]:
    """Return only windows that contain a five-in-a-row for either player."""
    return [w for w in sliding_five_windows(board) if w["pattern"] in ("BLACK_FIVE", "WHITE_FIVE")]


def find_patterns(board: List[List[int]], pattern_names: Iterable[str]) -> List[PatternInfo]:
    """Return windows matching one or more pattern names."""
    desired = set(pattern_names)
    return [w for w in sliding_five_windows(board) if w["pattern"] in desired]


def first_five_winner(board: List[List[int]]) -> int:
    """Return BLACK, WHITE, or 0 if no five-in-a-row exists."""
    for window in sliding_five_windows(board):
        if window["pattern"] == "BLACK_FIVE":
            return BLACK
        if window["pattern"] == "WHITE_FIVE":
            return WHITE
    return EMPTY


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
    sample = board_from_moves([
        (BLACK, 7, 7),
        (BLACK, 7, 8),
        (BLACK, 7, 9),
        (BLACK, 7, 10),
        (BLACK, 7, 11),
    ])
    winner = first_five_winner(sample)
    print("winner:", winner)
    for window in find_winning_lines(sample):
        print(window)


#weights
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
    "FIVE": 2000,
}

_STATIC_POSITION_WEIGHT_MATRIX = [
    [9 - max(abs(y - 7), abs(x - 7)) for x in range(BOARD_SIZE)]
    for y in range(BOARD_SIZE)
]


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


def static_position_weight(board: List[List[int]], y: int, x: int) -> int:
    """Return a simple static weight for a stone on the board."""
    if not (0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE):
        raise ValueError(f"Point out of range: {(y, x)}")
    if board[y][x] == EMPTY:
        return 0
    return _STATIC_POSITION_WEIGHT_MATRIX[y][x]


def board_pattern_threat_score(board: List[List[int]], player: int) -> int:
    """Return a direct threat score from the board patterns for the given player."""
    validate_board(board)
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")

    player_prefix = "BLACK" if player == BLACK else "WHITE"
    return sum(
        _pattern_bonus(w["pattern"])
        for w in sliding_five_windows(board)
        if w["pattern"].startswith(player_prefix)
    )


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


def static_position_weight_matrix(board: List[List[int]]) -> List[List[int]]:
    """Return a 15x15 static matrix of stone positional weights."""
    validate_board(board)
    return [
        [_STATIC_POSITION_WEIGHT_MATRIX[y][x] for x in range(BOARD_SIZE)]
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


def evaluate_move(
    board: List[List[int]],
    y: int,
    x: int,
    player: int,
    attack_weight: float = 0.6,
    defense_weight: float = 1.2,
    mode: str = "balanced",
) -> int:
    """Evaluate the utility of placing a stone at (y, x) for the given player.

    mode can be:
    - 'attack': maximize self's pattern strength
    - 'defense': maximize opponent's threat, defend high-risk areas
    - 'balanced': weighted combination

    attack_weight and defense_weight allow fine-tuning the relative
    importance of offensive and defensive considerations.

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
        return int(self_weight * attack_weight + opponent_weight * (defense_weight * 0.5))
    elif mode == "defense":
        return int(self_weight * (attack_weight * 0.5) + opponent_weight * defense_weight)
    else:
        return int(self_weight * attack_weight + opponent_weight * defense_weight)


def board_position_score(
    board: List[List[int]],
    player: int,
    use_position_weight: bool = True,
    use_pattern_weight: bool = True,
) -> int:
    """Compute a board score for the given player.

    - If use_position_weight is False, return the player's stone count.
    - If use_position_weight is True and use_pattern_weight is True, return the sum of
      pattern-derived candidate empty-point weights.
    - If use_position_weight is True and use_pattern_weight is False, return a simple
      static position-weighted score for the player's stones.
    """
    validate_board(board)
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")

    if not use_position_weight:
        return sum(1 for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] == player)

    if use_pattern_weight:
        return sum(
            point_pattern_weight(board, y, x, player=player)
            for y in range(BOARD_SIZE)
            for x in range(BOARD_SIZE)
            if board[y][x] == EMPTY
        )

    return sum(
        static_position_weight(board, y, x)
        for y in range(BOARD_SIZE)
        for x in range(BOARD_SIZE)
        if board[y][x] == player
    )


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