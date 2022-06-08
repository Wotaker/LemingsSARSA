import datetime
from time import time
from typing import Callable
import gym

import env.env_lemings
from env.levels import *
from logs.plotter import Plotter

from training.sarsa import run_sarsa
from training.q_learning import run_q_learning


ALGORITHM = 'q-learning'
PLOT_SAVE_PATH = f"logs\\plots\\plot_{ALGORITHM}.png"
LOGS_DIR_PATH = "logs\\logs_sarsa"
SEED = 303              # Starting seed
VERBOSE = False         # Renders the game states if set as True
LOG_EP_EVRY = 1
EPISODE_WINDOW=25


# ===== Environment Parameters =====
# ----------------------------------
LEVEL = lvl_small
STOCHASTIC_WIND = False
EXTRA_MOVES = False

MY_ENV = gym.make(
    "LemingsEnv-v1", 
    level=LEVEL,
    stochastic_wind=STOCHASTIC_WIND,
    extra_moves=EXTRA_MOVES
)

# ===== Learning Parameters =====
# -------------------------------
RUNS = 1
EPISODES = 3000
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.95
EXPERIMENT_RATE = 0.05


if __name__ == "__main__":
    
    # Initialize Plotter
    plotter = Plotter(
        logs_dir_path=LOGS_DIR_PATH, 
        save_plot_path=PLOT_SAVE_PATH,
        n_action_buckets=200,
        episodes_moving_average=EPISODE_WINDOW
    )

    # Choose agent
    agent: Callable = run_sarsa if ALGORITHM == "sarsa" else run_q_learning

    # Run experiments
    start = time()
    for i, run in enumerate(range(RUNS)):
        print(f"\n=== EXPERIMENT NR {i + 1} ===\n")

        agent(
            SEED + i,
            MY_ENV,
            LEVEL,
            LOGS_DIR_PATH,
            EPISODES,
            LOG_EP_EVRY,
            LEARNING_RATE,
            EXPERIMENT_RATE,
            DISCOUNT_FACTOR,
            verbose=False
        )
    
    end = time()
    
    # Make plots
    plotter.make_plot(show=False)

    # Print elapsed time
    print(f"\nElapsed time: {datetime.timedelta(seconds=(end - start))}")
