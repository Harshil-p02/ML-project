"""
The expert recorder.
"""

import msvcrt
import random
import gym
import numpy as np
import time
import os

# Increase max_episodes for car-racing
# DECREASE LINGER IF YOU CANNOT CONTROL THE CAR PROPERLY!!!

# ENV = "MountainCar-v0"
ENV = "CarRacing-v0"
BINDINGS = {
    # w
    119: np.array((0, 1, 0)),
    # a
    97: np.array((-1, 0, 0)),
    # s
    115: np.array((0, 0, 1)),
    # d
    100: np.array((1, 0, 0))
}
SHARD_SIZE = 2000
# SLEEP_TIME = 1/100
MAX_EPISODES = 12000
LINGER = 10


def run_recorder():
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """

    env = gym.make(ENV)
    env._max_episode_steps = MAX_EPISODES

    exit = False
    cnt = 0
    prev_action = None

    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []

    print("Welcome to the expert recorder")
    print("To record press either a or d to move the agent left or right.")
    print("Once you're finished press 1 to save the data.")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")

    while not exit:
        done = False
        _last_obs = env.reset()

        while not done:
            # time.sleep(SLEEP_TIME)
            env.render()

            # default action, do nothing
            action = np.array((0, 0, 0))
            if prev_action is not None and not np.all(prev_action == 0):
                cnt += 1
                action = prev_action
                if cnt >= LINGER:
                    cnt = 0
                    action = np.array((0, 0, 0))
            else:

                # if key is pressed, get that key
                if msvcrt.kbhit():
                    key_pressed = ord(msvcrt.getch())
                    print(key_pressed)

                    # pressed 1 -> exit
                    if key_pressed == 49:
                        exit = True
                    elif key_pressed in BINDINGS:
                        action = BINDINGS[key_pressed]

            # exit game loop
            if exit:
                print("ENDING")
                done = True
            else:
                obs, reward, done, info = env.step(action)
                prev_action = action
                sarsa = (_last_obs, action)
                _last_obs = obs
                sarsa_pairs.append(sarsa)

    print("SAVING")
    # Save out recording data.
    num_shards = int(np.ceil(len(sarsa_pairs) / SHARD_SIZE))
    print(len(sarsa_pairs))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
                shard_iter * SHARD_SIZE: min(
                    (shard_iter + 1) * SHARD_SIZE, len(sarsa_pairs))]


        np.save(f"{shard_iter}_{shard_suffix}", sarsa_pairs)


dir = os.getcwd()
run_recorder()