import traceback
import numpy as np

from logs.logger import Logger
from env.levels import LemingsLevel
from env.env_lemings import LemingsEnv
from training.common import init_Q, epsilon_greedy


def run_q_learning(
    seed: int,
    environment: LemingsEnv, 
    level: LemingsLevel, 
    logs_dir_path: str,
    episodes: int,
    log_ep_every: int=1,
    lr: float=0.5,
    epsilon: float=0.05,
    discount: float=0.95,
    decay: float=0.0,
    verbose: bool=False
):
    """
    Q-learning algorithm with logging

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
    log_ep_every : int, optional
        Logs episode stats every this number of episodes, by default 1
    lr : float, optional
        Learning rate, by default 0.5
    epsilon : float, optional
        Experiment rate, by default 0.05
    discount : float, optional
        Discount of return, by default 0.95
    decay : float, optional
        Decay in learning rate calculation, by default 0.0, meaning constant learning rate
    verbose : bool, optional
        If set as True, renders each state, by default False
    """

    # Set seed
    np.random.seed(seed)
    print(f"=== Running Q-Learning algorithm with seed {seed} ===\n")

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

    # Create and initialize Q table
    Q = init_Q(shape=(environment.get_q_shape(include_moves_done=False)), type="random")

    lr0 = lr
    total_action_counter = 0
    for ep in range(1, episodes + 1):

        if ep == 1 or ep % (episodes // 40) == 0:
            print(f"[Info] Episode nr {ep}")

        # Init environment in starting state
        state = environment.reset()
        (y, x), leming_id, moves_done = state

        if verbose:
            print("Env reset:")
            environment.render(space=True)

        # Loop until done
        action_counter = 0
        step = 1
        done = False
        while not done:
            
            if verbose:
                print(f"Step {step}")
            
            # Choose action to take
            action = epsilon_greedy(Q, (y, x, leming_id - 1), environment.action_space_size, epsilon)
            
            # Take action a, observe new state s' and reward r
            state_next, reward, done, info = environment.step(action)
            (y_next, x_next), leming_id_next, moves_done_next = state_next

                # update counter
            total_action_counter += 1
            action_counter += 1

                # log
            logger.logg_action(total_action_counter, info["fate"] == "rescued")
            logger.logg_episode(ep, info["fate"], action_counter, reward, every=log_ep_every)

                # render state if verbose
            
            if verbose: #  or (ep == EPISODES and MY_ENV.is_last_leming())
                print(f"State: {state_next}")
                print(f"Reward: {reward}")
                print(info)
                environment.render(space=True)
            
            # Update Q table
            q_prev = Q[y, x, leming_id - 1, action]
            try:
                Q[y, x, leming_id - 1, action] = (1 - discount) * q_prev +\
                    discount * (reward + lr * np.max(Q[y_next, x_next, leming_id_next - 1, :]))
            except IndexError as e:
                print(f"Episode {ep}, step {step}")
                print(f"Leming_id: {leming_id}")
                print(f"Leming_id_next: {leming_id_next}")
                traceback.print_exc()
                exit(1)
                
            # Go to next step
            y, x, leming_id, moves_done = y_next, x_next, leming_id_next, moves_done_next

            step += 1
        
        # Update Learning Rate
        lr = lr0 / (1.0 + decay * ep)

    # Save logs to csv files
    logger.save_logs()

    # Render learned lemings path
    path = []

    state = environment.reset()
    (y, x), leming_id, moves_done = state
    path.append((y, x))
    action = epsilon_greedy(Q, (y, x, leming_id - 1), environment.action_space_size, 0.0, train=False)
    state, reward, done, info = environment.step(action)
    (y, x), leming_id_next, moves_done_next = state
    path.append((y, x))

    while info["fate"] == "unknown":
        action = epsilon_greedy(Q, (y, x, leming_id - 1), environment.action_space_size, 0.0, train=False)
        state, reward, done, info = environment.step(action)
        (y, x), leming_id_next, moves_done_next = state
        path.append((y, x))
    
    environment.render(path=path)

    return
