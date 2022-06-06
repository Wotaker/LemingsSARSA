from functools import partial
from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Plotter():
    """Plotter is created after whole experiment, which may consists of multiple runs"""

    def __init__(self, 
        logs_dir_path: str, 
        save_plot_path: str,
        n_action_buckets: int=400, 
        episodes_moving_average: int=10,
        fresh_data: bool=True,
    ) -> None:

        self.logs_dir_path = logs_dir_path
        self.save_plot_path = save_plot_path
        self.n_action_buckets = n_action_buckets
        self.episodes_moving_average = episodes_moving_average
        self.fresh_data = fresh_data

        if self.fresh_data:
            self._clear_logs_dir()
    
    def _clear_logs_dir(self):
        
        # if input(f"Do you really want to clear {self.logs_dir_path} directory [y/n]? ").lower() != "y":
        #     return
        
        files = os.listdir(self.logs_dir_path)
        files = list(filter(lambda f_name: f_name[:5] in ["actio", "episo"], files))
        for f in files:
            os.remove(os.path.join(self.logs_dir_path, f))

    
    def _bucketify(self, max_actions: int, df: pd.DataFrame):
        buckets = np.round(np.linspace(0, max_actions, self.n_action_buckets + 1)).astype('int')
        df["ActionNr"] = pd.cut(df["ActionNr"], buckets, labels=buckets[1:])
        return df.groupby(["ActionNr"], as_index=False).max().fillna(method='ffill')
    
    def _moving_mean(self, df: pd.DataFrame):
        df["RescuedRatio"] = df["RescuedRatio"].rolling(self.episodes_moving_average, min_periods=1).mean()
        df["TornsRatio"] = df["TornsRatio"].rolling(self.episodes_moving_average, min_periods=1).mean()
        df["BoredomRatio"] = df["BoredomRatio"].rolling(self.episodes_moving_average, min_periods=1).mean()
        df["TimeInActions"] = df["TimeInActions"].rolling(self.episodes_moving_average, min_periods=1).mean()
        df["TotalReward"] = df["TotalReward"].rolling(self.episodes_moving_average, min_periods=1).mean()
        return df

    def combine_episodes(self):

        prefix = "episodes"
        index_column = "EpisodeNr"

        # Select appropriate files
        files = os.listdir(self.logs_dir_path)
        files = list(filter(lambda f_name: f_name[:len(prefix)] == prefix, files))

        # Collect appropriate dataframes to list
        runs = []
        for f in files:
            runs.append(pd.read_csv(os.path.join(self.logs_dir_path, f)))
        
        # Moving Mean
        runs = list(map(self._moving_mean, runs))

        # Extract important parameters
        columns = runs[0].columns
        n = runs[0].shape[0]

        # Calculate mean
        mean = np.mean(np.stack(runs), axis=0)
        df_mean = pd.DataFrame(data=mean, columns=columns)
        df_mean[index_column] = np.arange(start=1, stop=n+1, dtype=np.int32)
        df_mean.set_index(index_column, inplace=True)

        # Calculate std
        std = np.std(np.stack(runs), axis=0)
        df_std = pd.DataFrame(data=std, columns=columns)
        df_std[index_column] = np.arange(start=1, stop=n+1, dtype=np.int32)
        df_std.set_index(index_column, inplace=True)

        # Assign to proper atributes
        self.episodes_mean_df = df_mean
        self.episodes_std_df = df_std
    
    def combine_actions(self):
        
        prefix = "actions"
        index_column = "Idx"

        # Select appropriate files
        files = os.listdir(self.logs_dir_path)
        files = list(filter(lambda f_name: f_name[:len(prefix)] == prefix, files))

        # Collect appropriate dataframes to list
        runs = []
        for f in files:            
            runs.append(pd.read_csv(os.path.join(self.logs_dir_path, f)))
        max_actions = np.max(list(map(lambda df: df.tail(1)["ActionNr"], runs)))
        
        # Bucketify
        runs = list(map(partial(self._bucketify, max_actions), runs))

        # print(list(map(lambda en: en[1].to_csv(
        #     os.path.join(self.logs_dir_path, f"run_{en[0]}.csv")
        # ), enumerate(runs))))

        # Extract important parameters
        columns = runs[0].columns
        n = runs[0].shape[0]

        # Calculate mean
        mean = np.mean(np.stack(runs), axis=0)
        df_mean = pd.DataFrame(data=mean, columns=columns)
        df_mean[index_column] = np.arange(start=1, stop=n+1, dtype=np.int32)
        df_mean.set_index(index_column, inplace=True)

        # Calculate std
        std = np.std(np.stack(runs), axis=0)
        df_std = pd.DataFrame(data=std, columns=columns)
        df_std[index_column] = np.arange(start=1, stop=n+1, dtype=np.int32)
        df_std.set_index(index_column, inplace=True)

        # Assign to proper atributes
        self.actions_mean_df = df_mean
        self.actions_std_df = df_std
    
    def save_combined(self):

        # Save episodes mean and std to csv
        self.episodes_mean_df.to_csv(os.path.join(self.logs_dir_path, f"mean_episodes.csv"), float_format='%.4f')
        self.episodes_std_df.to_csv(os.path.join(self.logs_dir_path, f"std_episodes.csv"), float_format='%.4f')

        # Save actions mean and std to csv
        self.actions_mean_df.to_csv(os.path.join(self.logs_dir_path, f"mean_actions.csv"), float_format='%.4f')
        self.actions_std_df.to_csv(os.path.join(self.logs_dir_path, f"std_actions.csv"), float_format='%.4f')

    def _plot_I(self, ax: plt.Axes):
        
        measurements = self.actions_mean_df.shape[0]
        x = np.array(self.actions_mean_df["ActionNr"])
        mean = np.array(self.actions_mean_df["Rescued"])
        std = np.array(self.actions_std_df["Rescued"])
        
        alpha = 1 - 0.95
        z = t.ppf(1 - alpha / 2, measurements - 1)

        ci_low = mean - z * std / np.sqrt(measurements)
        ci_high = mean + z * std / np.sqrt(measurements)

        ax.plot(x, mean)
        ax.fill_between(x, ci_low, ci_high, alpha=0.2)
        ax.set_title("Rescued vs Time")
        ax.ticklabel_format(style="scientific", axis='both')
        ax.set_xlabel("Time [actions]")
        ax.set_ylabel("Total Rescued")

    def _plot_II(self, ax: plt.Axes):

        colors_dict = {
            "RescuedRatio": "tab:blue",
            "TornsRatio": "tab:green",
            "BoredomRatio": "tab:orange"
        }
        
        measurements = self.episodes_mean_df.shape[0]
        x = np.array(self.episodes_mean_df.index)

        for col in ["RescuedRatio", "TornsRatio", "BoredomRatio"]:
            mean = np.array(self.episodes_mean_df[col])
            std = np.array(self.episodes_mean_df[col])
            
            alpha = 1 - 0.95
            z = t.ppf(1 - alpha / 2, measurements - 1)

            ci_low = mean - z * std / np.sqrt(measurements)
            ci_high = mean + z * std / np.sqrt(measurements)

            ax.plot(x, mean, label=col, color=colors_dict[col])
            ax.fill_between(x, ci_low, ci_high, alpha=0.2, color=colors_dict[col])
        
        ax.set_title("Fate (rate) vs Time")
        ax.ticklabel_format(style="scientific", axis='both')
        ax.set_xlabel("Time [episodes]")
        ax.set_ylabel("Ratio")
        ax.legend()


    def _plot_III(self, ax: plt.Axes):
        
        measurements = self.episodes_mean_df.shape[0]
        x = np.array(self.episodes_mean_df.index)
        mean = np.array(self.episodes_mean_df["TimeInActions"])
        std = np.array(self.episodes_mean_df["TimeInActions"])
        
        alpha = 1 - 0.95
        z = t.ppf(1 - alpha / 2, measurements - 1)

        ci_low = mean - z * std / np.sqrt(measurements)
        ci_high = mean + z * std / np.sqrt(measurements)

        ax.plot(x, mean)
        ax.fill_between(x, ci_low, ci_high, alpha=0.2)
        ax.set_title("Episode Duration vs Episode")
        ax.ticklabel_format(style="scientific", axis='both')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Duration [actions]")

    def _plot_IV(self, ax: plt.Axes):
        
        measurements = self.episodes_mean_df.shape[0]
        x = np.array(self.episodes_mean_df.index)
        mean = np.array(self.episodes_mean_df["TotalReward"])
        std = np.array(self.episodes_mean_df["TotalReward"])
        
        alpha = 1 - 0.95
        z = t.ppf(1 - alpha / 2, measurements - 1)

        ci_low = mean - z * std / np.sqrt(measurements)
        ci_high = mean + z * std / np.sqrt(measurements)

        ax.plot(x, mean)
        ax.fill_between(x, ci_low, ci_high, alpha=0.2)
        ax.set_title("Total Reward vs Episode")
        ax.ticklabel_format(style="scientific", axis='both')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
    
    def make_plot(self, show: bool=False):
        
        # Combine the logs and calculate statistics
        self.combine_episodes()
        self.combine_actions()
        self.save_combined()

        # Make plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        self._plot_I(axes[0, 0])
        self._plot_II(axes[0, 1])
        self._plot_III(axes[1, 0])
        self._plot_IV(axes[1, 1])
        fig.tight_layout()
        
        plt.savefig(self.save_plot_path)
        print(f"[Info] Saved evaluation plot in {self.save_plot_path}")

        if show:
            plt.show()  
