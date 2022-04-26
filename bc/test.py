import gym
import torch
from model import MountainCarModel
import numpy as np


N_EPISODES = 5


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    episode_reward = 0
    step = 0

    state = env.reset()
    while True:
        # get action
        agent.eval()
        tensor_state = torch.from_numpy(state).float()
        print(tensor_state)
        tensor_action = agent(tensor_state)
        # print(tensor_action.data)
        print(int(tensor_action.data.argmax()))


        # a = tensor_action.detach().numpy()[0]
        a = int(tensor_action.data.argmax())

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward

model = MountainCarModel(2, 3, device="cpu")
model.load_state_dict(torch.load("mountaincar/models/model.pt"))


env = gym.make("MountainCar-v0")
ep_rewards = []

for ep in range(N_EPISODES):
    reward = run_episode(env, model, True)
    ep_rewards.append(reward)

print(f"Average reward: {sum(ep_rewards)/len(ep_rewards)}")