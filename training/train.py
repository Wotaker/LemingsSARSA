import datetime
from time import time
from typing import Callable
import gym

import env.env_lemings
from env.levels import *
from logs.plotter import Plotter

from training.sarsa import run_sarsa
from training.q_learning import run_q_learning
from training.basic import run_basic


# ===== Environment Parameters =====
# ----------------------------------
SEED = 666
LEVEL = lvl_chimney
STOCHASTIC_WIND = False
EXTRA_MOVES = False

MY_ENV = gym.make(
    "LemingsEnv-v1", 
    level=LEVEL,
    stochastic_wind=STOCHASTIC_WIND,
    extra_moves=EXTRA_MOVES
)


# ===== Learning Parameters ========
# ----------------------------------
ALGORITHM = 'q'         # choose from ['q', 'sarsa', 'basic']
EPISODES = 5000
LEARNING_RATE = 1.0
EXPERIMENT_RATE = 0.05
DISCOUNT_FACTOR = 0.95
DECAY = 0.0005


# ===== Logging Parameters =========
# ----------------------------------
RUNS = 10
PLOT_SAVE_PATH = f"results\\plots\\plot_{ALGORITHM}_{LEVEL.name}_{RUNS}_dec={DECAY}.png"
LOGS_DIR_PATH = "logs\\logs"
VERBOSE = False                 # Renders the game states if set as True
LOG_EP_EVRY = 1
EPISODE_WINDOW=25


if __name__ == "__main__":
    
    # Initialize Plotter
    plotter = Plotter(
        logs_dir_path=LOGS_DIR_PATH, 
        save_plot_path=PLOT_SAVE_PATH,
        n_action_buckets=200,
        episodes_moving_average=EPISODE_WINDOW
    )

    # Choose agent
    agent: Callable
    args = [SEED, MY_ENV, LEVEL, LOGS_DIR_PATH, EPISODES, VERBOSE]
    if ALGORITHM.lower() == "basic":
        agent = run_basic
    elif ALGORITHM.lower() == "sarsa":
        agent = run_sarsa
        args = args[:-1] + [LOG_EP_EVRY, LEARNING_RATE, EXPERIMENT_RATE, DISCOUNT_FACTOR, DECAY] + args[-1:]
    elif ALGORITHM.lower() in ["q", "q-learning", "q_learning", "qlearning"]:
        agent = run_q_learning
        args = args[:-1] + [LOG_EP_EVRY, LEARNING_RATE, EXPERIMENT_RATE, DISCOUNT_FACTOR, DECAY] + args[-1:]
    else:
        print("[Error] Unexisting agent selected! Choose from [basic, sarsa, q-learning]")

    # Run experiments
    start = time()
    for i, run in enumerate(range(RUNS)):

        print(f"\n=== EXPERIMENT NR {i + 1} ===\n")
        args[0] = SEED + i
        agent(*args)
    
    end = time()
    
    # Make plots
    plotter.make_plot(show=False)

    # Print elapsed time
    print(f"\nElapsed time: {datetime.timedelta(seconds=(end - start))}")
