from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import gym
from gym import spaces

from env.levels import LemingsLevel


gym.envs.registration.register(
    id='LemingsEnv-v1',
    entry_point='env.env_lemings:LemingsEnv'
)


EMPTY = 0
ROCKS = 1
TORNS = 2


class LemingsEnv(gym.Env):

    def __init__(
        self, level: LemingsLevel, 
        stochastic_wind: bool = False, 
        extra_moves: bool = False,
        reward_scale: float = 5.0
    ) -> None:

        self._level = level
        self._board_hight = level.board.shape[0]
        self._board_width = level.board.shape[1]
        self._winds = level.winds
        self._stochastic_wind = stochastic_wind
        self._reward_scale = reward_scale

        # dbg
        print(f"Board shape: ({self._board_hight, self._board_width})")

        # Observation space
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self._board_hight - 1, self._board_width - 1]), 
                dtype=np.int16
            ),
            "leming_id": spaces.Discrete(level.n_lemings),
            "moves_done": spaces.Discrete(level.moves_limit)
        })

        # Action space
        self.action_space_size = 6 if extra_moves else 2
        self.action_space = spaces.Discrete(self.action_space_size)

        self._action_dict = {
            0: (0, -1),     # left
            1: (0, 1),      # right
            2: (1, -1),     # left-down
            3: (1, 0),      # down
            4: (1, 1),      # right-down
            5: (0, 0)       # stay
        }

        self.symbols_dict = {
            "enter": " ● ",
            "exit": " X ",
            "rocks": " = ",
            "torns": " @ ",
            "wind": " ↑ ",
            "empty": "   "
        }
    
    def _try_move(self, move: Tuple[int, int]) -> bool:
        "Returns False if move caused lemings death, else True"

        new_row = self._pos[0] + move[0]
        new_col = self._pos[1] + move[1]

        # Landed off the board - do not move
        if new_row >= self._board_hight or new_row < 0:
            return True
        if new_col >= self._board_width or new_col < 0:
            return True
        
        # Landed on rocks - do not move
        if self._level.board[new_row, new_col] == ROCKS:
            return True
        
        # Landed on torns - do not move, signalize death
        if self._level.board[new_row, new_col] == TORNS:
            return False
        
        # Landed on empty space - move
        self._pos = (new_row, new_col)
        return True    

    def _chek_rescued(self):
        "Checks if the current leming has arived to the exit"

        return self._pos == (self._board_hight - 1, self._board_width - 1)
    
    def _next_leming(self) -> bool:
        """
        This method updates the leming_id, resets its position and returns True if there are
        any lemings left to rescue. If there are no other lemings left, it does nothing and 
        returns False.
        """
        
        if self._leming_id < self._level.n_lemings:
            self._leming_id += 1
            self._moves_done = 0
            self._pos = (0, 0)
            return True
        return False
    
    def reset(self) -> Tuple[Tuple[int, int], int, int]:

        self._pos = (0, 0)
        self._leming_id = 1
        self._moves_done = 0
        
        return self._pos, self._leming_id, self._moves_done
    
    def step(self, action: int) -> Tuple[Tuple[Tuple[int, int], int, int], float, bool, Dict[str, Any]]:
        
        assert action < self.action_space.__sizeof__(), \
            f"Illegal action! Choose from 0-{self.action_space.__sizeof__()}"
        
        # Make action
            # Try move
        if self._try_move(self._action_dict[action]) == False:      # leming is dead
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done + int(done)), -10, \
                done, {
                    "info": f"[Env Info] Leming {self._leming_id - int(not done)} was killed by a bunch of torns!",
                    "fate": "torns"
                }
        
            # Check if done
        if self._chek_rescued():                                    # leming rescued
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done + int(done)), 20, \
                done, {
                    "info": f"[Env Info] Leming {self._leming_id - int(not done)} was rescued!",
                    "fate": "rescued"
                }
        
            # Move by gravity
        if self._try_move((1, 0)) == False:                         # leming is dead
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done + int(done)), -10, \
                done, {
                    "info": f"[Env Info] Leming {self._leming_id - int(not done)} was killed by a bunch of torns!",
                    "fate": "torns"
                }
        
            # Check if done
        if self._chek_rescued():                                    # leming rescued
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done + int(done)), 20, \
                done, {
                    "info": f"[Env Info] Leming {self._leming_id - int(not done)} was rescued!",
                    "fate": "rescued"
                }
            
            # Move vertically by the wind
        lift = self._level.winds[self._pos[1]]
        if self._stochastic_wind and lift > 0:
            lift += np.random.choice([-1, 0, 1])
        for _ in range(lift):
            if self._try_move((-1, 0)) == False:                    # leming is dead
                done = not self._next_leming()
                return (self._pos, self._leming_id, self._moves_done + int(done)), -10, \
                    done, {
                        "info": f"[Env Info] Leming {self._leming_id - int(not done)} was killed by a bunch of torns!",
                        "fate": "torns"
                    }
            
            # Check if done
        if self._chek_rescued():                                    # leming rescued
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done + int(done)), 20, \
                done, {
                    "info": f"[Env Info] Leming {self._leming_id - int(not done)} was rescued!",
                    "fate": "rescued"
                }

            # Check moves limit
        self._moves_done += 1
        if self._moves_done >= self._level.moves_limit:
            done = not self._next_leming()
            return (self._pos, self._leming_id, self._moves_done), -self._level.moves_limit - 10, \
                done, {
                    "info": f"[Env Info] Moves limit exceeded, leming {self._leming_id - int(not done)} died of boredom!",
                    "fate": "boredom"
                }

            # Here, the leming managed to make the move safe and sound. He still has some moves left
        return (self._pos, self._leming_id, self._moves_done), 1, \
                False, {
                    "info": f"[Env Info] Leming {self._leming_id} moved and is still alive.",
                    "fate": "unknown"
                }
    
    def get_q_shape(self, include_moves_done: bool=False) -> Tuple:

        if not include_moves_done:
            return (
                self._board_hight,
                self._board_width,
                self._level.n_lemings,
                self.action_space_size
            )
        
        return (
            self._board_hight,
            self._board_width,
            self._level.n_lemings,
            self._level.moves_limit,
            self.action_space_size
        )

    
    def render(self, space=False) -> None:
        print("|" + "---" * self._board_width + "|")
        for row in range(self._board_hight):
            print("|", end="")
            for col in range(self._board_width):
                if self._pos == (row, col):
                    print(f" {self._leming_id} ", end="")
                elif row == 0 and col == 0:
                    print(self.symbols_dict["enter"], end="")
                elif row + 1 == self._board_hight and col + 1 == self._board_width:
                    print(self.symbols_dict["exit"], end="")
                elif self._level.board[row, col] == ROCKS:
                    print(self.symbols_dict["rocks"], end="")
                elif self._level.board[row, col] == TORNS:
                    print(self.symbols_dict["torns"], end="")
                elif self._level.winds[col] > 0:
                    print(self.symbols_dict["wind"], end="")
                else:
                    print(self.symbols_dict["empty"], end="")
            print("|")
        print("|" + "---" * self._board_width + "|")
        if space:
            print()

"""
|---------------|
| ●     ↑  ↑    |
| 2  @  ↑  ↑    |
|========= ↑    |
|       ↑  ↑  X |
|---------------|
"""

