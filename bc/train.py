import os
import gym
import torch
from model import MountainCarModel
import numpy as np


DATA_DIR = "mountaincar/"
N_EPOCHS = 100


def process_data(data_dir):
    states, actions = [], []
    shards = [x for x in os.listdir(data_dir) if x.endswith('.npy')]
    print(f"Processing shards: {shards}")
    for shard in shards:
        shard_path = os.path.join(data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            shard_states, unprocessed_actions = zip(*data)
            shard_states = [x.flatten() for x in shard_states]

            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    print(f"Processed with {len(states)} pairs")

    # print(states[:10])
    # print(actions[:10])
    # print(actions)

    return states, actions

def train(states, actions):
    model = MountainCarModel(2, 3, device="cpu")
    model.to("cpu")

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    enc_actions = np.zeros((len(actions), 3))
    for i in range(len(actions)):
        if actions[i] == 0:
            enc_actions[i][0] = 1
        elif actions[i] == 1:
            enc_actions[i][1] = 1
        else:
            enc_actions[i][2] = 1

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(states), torch.from_numpy(enc_actions))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)

    for epoch in range(N_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            print("Label:", labels)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), "mountaincar/models/model.pt")


def main():
    states, actions = process_data(DATA_DIR)

    # env = gym.make("MountainCar-v0")
    # env._max_episode_steps = 1200

    train(states, actions)

main()


