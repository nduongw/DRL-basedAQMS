import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, actionSpace, observationSpace):
        super().__init__()
        
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(338, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.actionSpace)
        
    def forward(self, input):
        flattenLayer = self.flatten(input)
        dense1 = F.relu(self.fc1(flattenLayer))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        output = self.fc4(dense3)
        prob = F.softmax(output)
        
        return prob