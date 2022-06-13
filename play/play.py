import gym

import env.env_lemings
from env.levels import *

my_env = gym.make(
    "LemingsEnv-v1", 
    level=lvl_debug,
    stochastic_wind=False,
    extra_moves=False
)


action_dict = {
    0: "Left",              # left
    1: "Right",             # right
    2: "Left-down",         # left-down
    3: "Down",              # down
    4: "Right-down",        # right-down
    5: "Stay"               # stay
}

def print_help():
    print("Press 'q' to quit, 'h' to display this message.")
    print("Your moves:")
    for key, val in action_dict.items():
        print(f"  {val} - {key}")
    print()


if __name__ == "__main__":
    print("\n === Rescue Lemings ===\n")
    print_help()
    my_env.reset()
    print("Env reset:")
    my_env.render(space=True)

    i = 1
    done = False
    while not done:
        print(f"Step {i}")
        act = input("Select action: ")

        if act.lower() == 'q':
            break
        elif act.lower() == 'h':
            print_help()
        else:
            s, r, done, info = my_env.step(int(act))
            print(f"State: {s}")
            print(f"Reward: {r}")
            print(info)
            my_env.render(space=True)
            i += 1