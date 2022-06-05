from logs.plotter import Plotter


if __name__ == "__main__":
    plotter = Plotter(
        logs_dir_path="logs\\logs_sarsa",
        save_plot_path="logs\\plots\\plot_offline.png",
        n_action_buckets=200,
        episodes_moving_average=25,
        fresh_data=False
    )

    plotter.make_plot(show=False)
