import traceback
import numpy as np

from logs.logger import Logger
from env.levels import LemingsLevel
from env.env_lemings import LemingsEnv


def run_basic(
    seed: int,
    environment: LemingsEnv,
    level: LemingsLevel,
    logs_dir_path: str,
    episodes: int,
    verbose: bool=False
):
    """
    Basic agent that randomly selects action from environments action space

    Parameters
    ----------
    seed : int
        Seed to reproduce results
    environment : LemingsEnv
        Environment that the agent interacts with
    level : LemingsLevel
        Lemings level on which the action takes place 
    logs_dir_path : str
        The diroctory where the logs are saved. Should be empty, otherwise the data could be lost
    episodes : int
        Duration of the training in episodes
    verbose : bool, optional
        If set as True, renders each state, by default False
    """

    # Set seed
    np.random.seed(seed)
    print(f"=== Running Basic algorithm with seed {seed} ===\n")

    # Initialize logger
    logger = Logger(
        level.n_lemings,
        lemings_history_size=5,
        logs_dir_path=logs_dir_path, 
        params={
            "lvl": level.name,
            "nEp": episodes,
            "nLemings": level.n_lemings,
            "movesLim": level.moves_limit,
            "seed": seed
        }
    )

    total_action_counter = 0
    for ep in range(1, episodes + 1):

        if ep == 1 or ep % 25 == 0:
            print(f"[Info] Episode nr {ep}")

        environment.reset()
        
        if verbose:
            print("Env reset:")
            environment.render(space=True)

        action_counter = 0
        step = 1
        done = False
        while not done:
            
            if verbose:
                print(f"Step {step}")

            act = environment.action_space.sample()
            s, r, done, info = environment.step(act)

            total_action_counter += 1
            action_counter += 1
            
            logger.logg_action(total_action_counter, info["fate"] == "rescued")
            logger.logg_episode(ep, info["fate"], action_counter, r)

            if verbose:
                print(f"State: {s}")
                print(f"Reward: {r}")
                print(info)
                environment.render(space=True)

            step += 1
    
    # Save logs to csv files
    logger.save_logs()
