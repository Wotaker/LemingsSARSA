import numpy as np
import pandas as pd
import os


class Logger():
    """Logger is created per run"""

    def __init__(self, n_lemings, lemings_history_size=20, logs_dir_path=f"logs{os.sep}logs", params={}):
        
        self.n_lemings = n_lemings
        self.lemings_history_size = lemings_history_size
        self.logs_dir_path = logs_dir_path
        self.params = params

        self.lemings_left = n_lemings
        self.last_episode = 0

        # Stores data for accumulative rescued DataFrame
        # key: action number
        # value: Tuple[seed, total rescued]
        self.dict_acc_rescued = dict()
        self.rescued = 0
        
        # Stores data for episodes info DataFrame
        # key: episode number
        # value: Tuple[seed, total rescued, total torns, total boredom, time in actions, total reward]
        self.dict_episodes = dict()
    
    def logg_action(self, action_nr: np.int64, rescued: bool):
        
        if rescued:
            self.rescued += 1
            self.dict_acc_rescued[self.rescued] = (action_nr, self.rescued)

    def logg_episode(self, episode_nr: np.int32, fate: str, actions: int, reward: int):

        # Reset lemings left counter if new episode
        if episode_nr > self.last_episode:
            self.lemings_left = self.n_lemings
            self.last_episode = episode_nr
        
        # Decrement lemings left counter on rescue or death
        self.lemings_left -= int(fate != "unknown")
        
        # Bool flag if logg or not the fate
        logg_fate = self.lemings_left <= self.lemings_history_size

        # Determine the fate
        rescued = int(fate == "rescued" and logg_fate)
        torns = int(fate == "torns" and logg_fate)
        boredom = int(fate == "boredom" and logg_fate)

        # On existing episode
        if episode_nr in self.dict_episodes.keys():
            outdated = self.dict_episodes[episode_nr]
            self.dict_episodes[episode_nr] = (
                outdated[0] + rescued,
                outdated[1] + torns,
                outdated[2] + boredom,
                actions,
                outdated[4] + reward
            )
            return
        
        # On new episode
        self.dict_episodes[episode_nr] = (rescued, torns, boredom, actions, reward)
    
    def save_logs(self):

        params_str = "".join([f"_{key}={str(val)}" for key, val in self.params.items()])

        # Save accumulative rescued logs
        df_acc_rescued = pd.DataFrame.from_dict(
            data=self.dict_acc_rescued,
            columns=["ActionNr", "Rescued"],
            orient='index'
        )
        df_acc_rescued.index.name = "Idx"
        df_acc_rescued.to_csv(os.path.join(self.logs_dir_path, f"actions{params_str}.csv"))

        # Save episodes info logs
        df_episode = pd.DataFrame.from_dict(
            data=self.dict_episodes,
            columns=["RescuedRatio", "TornsRatio", "BoredomRatio", "TimeInActions", "TotalReward"],
            orient='index'
        )
        # df_episode.index += 1
        df_episode.index.name = "EpisodeNr"
        total_fate = df_episode["RescuedRatio"] + df_episode["TornsRatio"] + df_episode["BoredomRatio"]
        df_episode["RescuedRatio"] = df_episode["RescuedRatio"] / total_fate
        df_episode["TornsRatio"] = df_episode["TornsRatio"] / total_fate
        df_episode["BoredomRatio"] = df_episode["BoredomRatio"] / total_fate
        df_episode.to_csv(os.path.join(self.logs_dir_path, f"episodes{params_str}.csv"))
