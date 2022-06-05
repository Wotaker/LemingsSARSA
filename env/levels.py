import numpy as np


class LemingsLevel():

    def __init__(
        self, name: str,
        board: np.ndarray, 
        winds: np.ndarray,
        n_lemings: int = 1,
        moves_limit: int = 100
    ) -> None:
        self.name = name
        self.board = board
        self.winds = winds
        self.n_lemings = n_lemings
        self.moves_limit = moves_limit
        return
    
    def __str__(self) -> str:
        return "Board Representation"


lvl_small = LemingsLevel(
    name="Small",
    board=np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    winds=np.array(
        [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0]
    ),
    n_lemings=100,
    moves_limit=80
)

lvl_debug = LemingsLevel(
    name="Dbg",
    board=np.array([
        [0, 0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0]
    ]),
    winds=np.array(
        [0, 0, 0, 0, 0, 0]
    ),
    n_lemings=10,
    moves_limit=15
)
