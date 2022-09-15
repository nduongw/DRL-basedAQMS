from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class DQNDenseModel(nn.Module):
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
    
class DQNCNNModel(nn.Module):
    def __init__(self, actionSpace, observationSpace, device):
        super().__init__()
        
        self.device = device
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        
        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=4, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        
        self.fc1 = nn.Linear(8 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.actionSpace)
        
    def forward(self, obs):
        con1 = F.relu(self.conv1(obs))
        # print(con1.shape)
        con2 = F.relu(self.conv2(con1))
        # print(con2.shape)
        flatten = self.flatten(con2)
        # print(flatten.shape)
        dense1 = F.relu(self.fc1(flatten))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        output = F.relu(self.fc4(dense3))
        
        return output
    
class DQNCNNModel2(nn.Module):
    def __init__(self, actionSpace, observationSpace, device):
        super().__init__()
        
        self.device = device
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        
        self.flatten = nn.Flatten()
        
        self.coverConv1 = nn.Conv2d(in_channels=1,out_channels=2, kernel_size=5)
        self.coverConv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        
        self.posConv1 = nn.Conv2d(in_channels=1,out_channels=2, kernel_size=5)
        self.posConv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        
        self.fc1 = nn.Linear(4 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.actionSpace)
        
    def forward(self, obs):
        coverConv1 = F.relu(self.coverConv1(obs[:,0:1,:,:]))
        coverConv2 = F.relu(self.coverConv2(coverConv1))

        posConv1 = F.relu(self.coverConv1(obs[:,1:,:,:]))
        posConv2 = F.relu(self.coverConv2(posConv1))
        # print(coverConv2.shape)
        # print(posConv2.shape)
        
        outputConv = torch.stack([coverConv2, posConv2], dim=1)
        # print(con1.shape)
        # print(con2.shape)
        # print('Output shape: ', outputConv.shape)
        flatten = self.flatten(outputConv)
        # print('Flatten shape: ', flatten.shape)
        dense1 = F.relu(self.fc1(flatten))
        dense2 = F.relu(self.fc2(dense1))
        output = F.relu(self.fc3(dense2))
        
        return output
    
if __name__ == "__main__":
    inputMatrix = torch.rand([1, 2, 21, 21])
    model = DQNDenseModel(3, [2, 13, 13], 'cpu')
    model2 = DQNCNNModel(2, [2, 13, 13], 'cpu')
    model3 = DQNCNNModel2(2, [2, 13, 13], 'cpu')
    output = model(inputMatrix)
    output2 = model2(inputMatrix)
    output3 = model3(inputMatrix)
    # print(output.shape)
    print(output3.shape)
