import datetime
from time import time
from typing import Tuple
import traceback
import numpy as np
import gym

import env.env_lemings
from env.levels import *
from logs.logger import Logger
from logs.plotter import Plotter

PLOT_SAVE_PATH = "logs\\plots\\plot_sarsa.png"
LOGS_DIR_PATH = "logs\\logs_sarsa"
SEED = 421              # Starting seed
VERBOSE = False         # Renders the game states if set as True
LOG_EP_EVRY = 500


# === Environment Parameters ===
LEVEL = lvl_chimney
STOCHASTIC_WIND = False
EXTRA_MOVES = False

MY_ENV = gym.make(
    "LemingsEnv-v1", 
    level=LEVEL,
    stochastic_wind=STOCHASTIC_WIND,
    extra_moves=EXTRA_MOVES
)

# === Learning Parameters ===
RUNS = 1
EPISODES = 5000
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.95
EXPERIMENT_RATE = 0.05


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


def run_sarsa(seed):

    # Set seed
    np.random.seed(seed)

    # Initialize logger
    logger = Logger(
        LEVEL.n_lemings,
        lemings_history_size=5,
        logs_dir_path=LOGS_DIR_PATH, 
        params={
            "lvl": LEVEL.name,
            "nEp": EPISODES,
            "nLemings": LEVEL.n_lemings,
            "movesLim": LEVEL.moves_limit,
            "seed": seed
        }
    )

    # Create and initialize Q table
    Q = init_Q(shape=(MY_ENV.get_q_shape(include_moves_done=False)), type="random")
    # if seed == SEED:
    #     print(f" = = = Q szhape: {Q.shape}")

    total_action_counter = 0
    for ep in range(1, EPISODES + 1):

        if ep == 1 or ep % LOG_EP_EVRY == 0:
            print(f"[Info] Episode nr {ep}")

        # Init environment in starting state
        state = MY_ENV.reset()
        (y, x), leming_id, moves_done = state

        if VERBOSE:
            print("Env reset:")
            MY_ENV.render(space=True)

        # Choose first action to take
        action = epsilon_greedy(Q, (y, x, leming_id - 1), MY_ENV.action_space_size, EXPERIMENT_RATE)

        # Loop until done
        action_counter = 0
        step = 1
        done = False
        while not done:
            
            if VERBOSE:
                print(f"Step {step}")
            
            # Take action a, observe new state s' and reward r
            state_next, reward, done, info = MY_ENV.step(action)
            (y_next, x_next), leming_id_next, moves_done_next = state_next

                # update counter
            total_action_counter += 1
            action_counter += 1

                # log
            logger.logg_action(total_action_counter, info["fate"] == "rescued")
            logger.logg_episode(ep, info["fate"], action_counter, reward)

                # render state if verbose
            if VERBOSE:
                print(f"State: {state_next}")
                print(f"Reward: {reward}")
                print(info)
                MY_ENV.render(space=True)

            # Choose action a' from s' using e-greedy policy based on Q
            action_next = epsilon_greedy(
                Q, (y_next, x_next, leming_id_next - 1), MY_ENV.action_space_size, EXPERIMENT_RATE
            )
            
            # Update Q table
            q_prev = Q[y, x, leming_id - 1, action]
            try:
                Q[y, x, leming_id - 1, action] = (1 - DISCOUNT_FACTOR) * q_prev +\
                    DISCOUNT_FACTOR * (reward + LEARNING_RATE * Q[y_next, x_next, leming_id_next - 1, action_next])
            except IndexError as e:
                print(f"Episode {ep}, step {step}")
                print(f"Leming_id: {leming_id}")
                print(f"Leming_id_next: {leming_id_next}")
                traceback.print_exc()
                exit(1)
                
            
            # Go to next step
            y, x, leming_id, moves_done = y_next, x_next, leming_id_next, moves_done_next
            action = action_next

            step += 1

    # Save logs to csv files
    logger.save_logs()



if __name__ == "__main__":
    
    # Initialize Plotter
    plotter = Plotter(
        logs_dir_path=LOGS_DIR_PATH, 
        save_plot_path=PLOT_SAVE_PATH,
        n_action_buckets=200,
        episodes_moving_average=25
    )

    # Run experiments
    start = time()
    for i, run in enumerate(range(RUNS)):
        print(f"\n=== EXPERIMENT NR {i + 1} ===\n")
        run_sarsa(SEED + i)
    end = time()
    
    # Make plots
    plotter.make_plot(show=False)

    # Print elapsed time
    print(f"\nElapsed time: {datetime.timedelta(seconds=(end - start))}")
