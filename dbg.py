import numpy as np
import gym

import env.env_lemings
from env.levels import *
from logs.logger import Logger
from logs.plotter import Plotter

RUNS = 1               # How many experiments to run
EPISODES = 1000           # How many episodes in each run
SEED = 44               # Starting seed
VERBOSE = False         # Renders the game states if set as True
LEVEL = lvl_debug       # Choose a level to run the experiments

PLOT_SAVE_PATH = "logs\\plots\\plot_dbg.png"
LOGS_DIR = "logs\\logs_test"


my_env = gym.make("LemingsEnv-v1", level=LEVEL)


def run_experiment(seed):

    # Set seed
    np.random.seed(seed)

    # Initialize logger
    logger = Logger(
        LEVEL.n_lemings,
        lemings_history_size=5,
        logs_dir_path=LOGS_DIR, 
        params={
            "lvl": LEVEL.name,
            "nEp": EPISODES,
            "nLemings": LEVEL.n_lemings,
            "movesLim": LEVEL.moves_limit,
            "seed": seed
        }
    )

    total_action_counter = 0
    for ep in range(1, EPISODES + 1):

        if ep == 1 or ep % 25 == 0:
            print(f"[Info] Episode nr {ep}")

        my_env.reset()
        
        if VERBOSE:
            print("Env reset:")
            my_env.render(space=True)

        action_counter = 0
        step = 1
        done = False
        while not done:
            
            if VERBOSE:
                print(f"Step {step}")

            act = my_env.action_space.sample()
            s, r, done, info = my_env.step(act)

            total_action_counter += 1
            action_counter += 1

            # dbg
            # if info["fate"] == "rescued":
            #     print("RESCUED!")
            
            logger.logg_action(total_action_counter, info["fate"] == "rescued")
            logger.logg_episode(ep, info["fate"], action_counter, r)

            if VERBOSE:
                print(f"State: {s}")
                print(f"Reward: {r}")
                print(info)
                my_env.render(space=True)

            step += 1
    
    # Save logs to csv files
    logger.save_logs()


if __name__ == "__main__":

    # Initialize Plotter
    plotter = Plotter(
        logs_dir_path=LOGS_DIR, 
        save_plot_path=PLOT_SAVE_PATH,
        n_action_buckets=200,
        episodes_moving_average=25
    )

    # Run experiments
    for i, run in enumerate(range(RUNS)):
        print(f"\n=== EXPERIMENT NR {i + 1} ===\n")
        run_experiment(SEED + i)
    
    # Make plots
    plotter.make_plot()
