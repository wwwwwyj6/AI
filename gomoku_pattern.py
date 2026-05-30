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
