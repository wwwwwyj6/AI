"""Gomoku board evaluator used by search leaf nodes.

Provides `evaluate(board, player, ...)` which returns a signed score:
positive = `player` is better, negative = opponent is better.

This module reuses names and functions from gomoku_weights.py and gomoku_pattern.py
so variable names are consistent with the rest of the project.
"""
from typing import List

from gomoku_pattern import BLACK, WHITE, EMPTY, BOARD_SIZE
from gomoku_weights import (
    validate_board,
    board_position_score,
    board_from_moves,
)


def evaluate(
    board: List[List[int]],
    player: int,
    use_position_weight: bool = True,
    use_pattern_weights: bool = True,
) -> int:
    """Evaluate the given 15x15 `board` for `player`.

    Args:
        board: 15x15 matrix with values EMPTY/BLACK/WHITE.
        player: BLACK or WHITE - the side for which the score is positive.
        use_position_weight: if True, use pattern-based position weights (board_position_score);
            if False, fall back to simple stone-count difference.
        use_pattern_weights: when using position weights, pass this to board_position_score to
            toggle pattern-weighted evaluation vs simple count.

    Returns:
        Signed integer score: positive means `player` advantage; negative means opponent advantage.
    """
    validate_board(board)
    if player not in (BLACK, WHITE):
        raise ValueError("player must be BLACK or WHITE")

    opponent = WHITE if player == BLACK else BLACK

    if use_position_weight:
        self_score = board_position_score(board, player, use_pattern_weights=use_pattern_weights)
        opp_score = board_position_score(board, opponent, use_pattern_weights=use_pattern_weights)
        return int(self_score - opp_score)

    # fallback: stone-count difference
    player_count = sum(1 for r in board for c in r if c == player)
    opp_count = sum(1 for r in board for c in r if c == opponent)
    return player_count - opp_count


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

    print("evaluate (position weights) for BLACK:", evaluate(board, BLACK, use_position_weight=True))
    print("evaluate (position weights) for WHITE:", evaluate(board, WHITE, use_position_weight=True))
    print("evaluate (count only) for BLACK:", evaluate(board, BLACK, use_position_weight=False))
    print("evaluate (count only) for WHITE:", evaluate(board, WHITE, use_position_weight=False))
