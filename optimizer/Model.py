import torch.nn as nn
import torch.nn.functional as F
import torch

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
        # print(flattenLayer.shape)
        dense1 = F.relu(self.fc1(flattenLayer))
        # print(dense1.shape)
        dense2 = F.relu(self.fc2(dense1))
        # print(dense2.shape)
        dense3 = F.relu(self.fc3(dense2))
        # print(dense3.shape)
        output = self.fc4(dense3)
        # print(output)
        
        return output
    

if __name__ == "__main__":
    inputMatrix = torch.rand([10, 2, 13, 13])
    model = DQNModel(3, [2, 13,13])
    output = model(inputMatrix)
    print(output)