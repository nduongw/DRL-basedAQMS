import torch.nn as nn
import torch.nn.functional as F
import torch

class DQNModel(nn.Module):
    def __init__(self, actionSpace, observationSpace, device):
        super().__init__()
        
        self.device = device
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(338, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.actionSpace)
        
    def forward(self, obs):
        flattenLayer = self.flatten(obs)
        dense1 = F.relu(self.fc1(flattenLayer))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        output = self.fc4(dense3)
        
        return output
    

if __name__ == "__main__":
    inputMatrix = torch.rand([2, 13, 13])
    model = DQNModel(3, [2, 13,13])
    output = model(inputMatrix, 0)
    print(output)