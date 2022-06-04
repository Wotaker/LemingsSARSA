from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logs.plotter import Plotter

N_BUCKETS = 4
N_ROLLING = 3

def main():
    plotter = Plotter(logs_dir_path="logs\\logs_dbg", save_plot_path="logs\\plots\\dbg_plot.png")
    plotter.make_plot()

def buckets():

    # Raw Actions
    df = pd.DataFrame(data={
        "ActionNr": [67,118,153,202,297,340,437,501],
        "Rescued": [1,2,3,4,5,6,7,8]
    })
    print(df)

    list_df = list(map(partial(bucketify, 600), [df]))
    df = list_df[0]
    print(df)

    plt.plot(df["ActionNr"], df["Rescued"])
    plt.show()

def bucketify(max_actions: int, df: pd.DataFrame):
    buckets = np.round(np.linspace(0, max_actions, N_BUCKETS)).astype('int')
    df["ActionNr"] = pd.cut(df["ActionNr"], buckets, labels=buckets[1:])
    return df.groupby(["ActionNr"], as_index=False).max()

def moving_mean(df: pd.DataFrame):
        df["TimeInActions"] = df["TimeInActions"].rolling(N_ROLLING, min_periods=1).mean()
        df["TotalReward"] = df["TotalReward"].rolling(N_ROLLING, min_periods=1).mean()

def rolling():
    print("N_ROLLING:", N_ROLLING)
    df = pd.read_csv("logs\\logs_dbg\\episodes_lvl=Small_nEp=250_nLemings=100_movesLim=80_seed=43.csv")
    print(df)
    print()

    # df["TimeInActions"] = df["TimeInActions"].rolling(N_ROLLING, min_periods=1).mean()
    # df["TotalReward"] = df["TotalReward"].rolling(N_ROLLING, min_periods=1).mean()
    moving_mean(df)
    print(df)

if __name__ == "__main__":
    main()
    # rolling()
