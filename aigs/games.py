# games.py
#   aigs games
# by: Noah Syrkis

# imports
from aigs.types import State, Env
import numpy as np


def connect_four_test(v):
    if len(v) < 4:
        return False

    for i in range(len(v) - 3):
        if v[i] and v[i + 1] and v[i + 2] and v[i + 3]:
            return True
    else:
        return False


# connect four
class ConnectFour(Env):
    def init(self) -> State:
        board = np.zeros((6, 7), dtype=int)
        legal = board[0] == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # place piece
        board = state.board.copy()
        col = board[:, action]  # <- a vector
        assert col[0] == 0
        row = np.where(col == 0)[0][-1]
        board[row, action] = 1 if state.maxim else -1

        # detect winner
        mask = board == (1 if state.maxim else -1)
        rows = [row for row in mask]
        cols = [col for col in mask.T]
        r_diags = [mask.diagonal(i) for i in range(-6, 7)]
        l_diags = [mask.T.diagonal(i) for i in range(-7, 6)]
        lst = [connect_four_test(v) for v in rows + cols + r_diags + l_diags]

        winner = True in lst
        legal = board[0] == 0
        point = (1 if state.maxim else -1) if winner else 0

        return State(
            board=board,
            legal=legal,
            ended=not legal.any() or winner,
            point=point,
            maxim=not state.maxim,
        )


# tic tac toe
class TicTacToe(Env):
    def init(self) -> State:
        board = np.zeros((3, 3), dtype=int)
        legal = board.flatten() == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # make your move
        board = state.board.copy()
        assert board[action // 3, action % 3] == 0, f"Invalid move: {action}"
        board[action // 3, action % 3] = 1 if state.maxim else -1

        # was it a winning move?
        mask = board == (1 if state.maxim else -1)
        winner: bool = (
            mask.all(axis=1).any()  # |
            or mask.all(axis=0).any()  # —
            or mask.trace() == 3  # \
            or mask.T.trace() == 3  # /
        )

        # return the next state
        return State(
            board=board,
            legal=board.flatten() == 0,  # empty board positions
            ended=(board != 0).all() or winner,
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )
