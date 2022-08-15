from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

writer = SummaryWriter('runs/plot_model_graph')
img = np.ones([100, 50])
for i in range(50):
    img[i, i] = 0
    img[2 * i, i] = 166
writer.add_image('Image', img, 0, dataformats='HW')

# class DQNModel(nn.Module):
#     def __init__(self, actionSpace, observationSpace, device):
#         super().__init__()
        
#         self.device = device
#         self.actionSpace = actionSpace
#         self.observationSpace = observationSpace
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(338, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, self.actionSpace)
        
#     def forward(self, obs):
#         # print('Flatten dimension', flattenDim)
#         # print('Input shape: ', obs.shape)
#         flattenLayer = self.flatten(obs)
#         dense1 = F.relu(self.fc1(flattenLayer))
#         # print(dense1.shape)
#         dense2 = F.relu(self.fc2(dense1))
#         # print(dense2.shape)
#         dense3 = F.relu(self.fc3(dense2))
#         # print(dense3.shape)
#         output = self.fc4(dense3)
#         # print(output)
        
#         return output
    
# model = DQNModel(2, 338, 'cpu')
# inputShape = torch.rand([1, 2, 13, 13])
# writer.add_graph(model, inputShape)
writer.close()
    