import torch
import torch.nn as nn
import torch.nn.functional as F

class MountainCarModel(nn.Module):

    def __init__(self, inp_size, out_size, device):
        super(MountainCarModel, self).__init__()

        self.fc1 = nn.Linear(inp_size, 128, device=device)
        self.fc2 = nn.Linear(128, 128, device=device)
        # self.fc3 = nn.Linear(256, 64, device=device)
        self.logits = nn.Linear(128, out_size, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.logits(x)

        return x