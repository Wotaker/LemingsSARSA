from typing import Tuple
import numpy as np


def init_Q(shape: Tuple, type: str="equal") -> np.ndarray:
    """
    Initializes the Q table

    Parameters
    ----------
    shape : Tuple
        The shape of the Q table. Shoul have at least 2 dimensions, one for
        states and one for actions. The last dimension is reserved for actions
    type : str, optional
        The type of initialization. Choose from ["equal", "random"], by default "equal"

    Returns
    -------
    np.ndarray
        The initialized Q table
    """

    Q = -10000 * np.ones(shape)

    if type == "equal":
        pass
    if type == "random":
        Q = Q + np.random.random(shape)
    else:
        print("[Warning] Unsuported Q initialization type! Initializing with ones")
    
    # Condition that Q[terminal, :] = 0
    if len(shape) == 4:
        Q[-1, -1, -1, :] = 0
    elif len(shape) == 5:
        Q[-1, -1, -1, :, :] = 0
    
    return Q


def epsilon_greedy(Q: np.ndarray, s: Tuple, n_actions:int,  epsilon: float, train: bool=True) -> int:
    """
    Chooses action from Q given current state s with e-greedy policy.
    If train set to false, than just selects best action from Q(s)

    Parameters
    ----------
    Q : np.ndarray
        Q table
    s : Tuple
        current state
    n_actions : int
        number of actions to choose from
    epsilon : float
        experiment rate parameter
    train : bool, optional
        if set as false, always return best action possible (do not consider epsilon), by default True

    Returns
    -------
    int
        next action to take
    """

    if (not train) or np.random.rand() > epsilon:
        if len(s) == 3:
            action = np.argmax(Q[s[0], s[1], s[2], :])
        else:
            action = np.argmax(Q[s[0], s[1], s[2], s[3], :])
    else:
        action = np.random.randint(0, n_actions)
    return action
