import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import random
from config import Config
from utils.TensorBoardUtils import *

class Agent:
    def __init__(self, model, optimizer, memory, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
        self.lossFunction = nn.MSELoss()
        self.device = device    
            
    def train(self, index, writer):
        totalLoss = 0
        # print(self.memory.size()
        for _ in range(min(int(self.memory.size() / Config.batchSize), 50)):
            s, a, r, sPrime = self.memory.sample()
            
            output = self.model(s)
            # print('Output Q values:') 
            # print(output)
            actualQ = output.gather(1, a.type(torch.int64))
            # print('Actual Q: ') 
            # print(actualQ)
            # print('Output next Q')
            maxQPrime = self.model((sPrime)).max(1)[0].unsqueeze(1)
            
            # print('Max Q: ') 
            # print(maxQPrime)
            targetQ = r + Config.gamma * maxQPrime
            # print('Target Q: ') 
            # print(targetQ)
            
            loss = self.lossFunction(targetQ, actualQ)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            totalLoss += loss
        writer.add_scalar("Loss", totalLoss / min(int(self.memory.size() / Config.batchSize), 50), index)
        return totalLoss / min(int(self.memory.size() / Config.batchSize), 50)
            
    def getAction(self, observation, epsilon):
        observation = torch.from_numpy(observation)
        observation = T.Resize((13, 13))(observation).type(torch.float).to(self.device).unsqueeze(0)

        qOut = self.model(observation)
        # print(qOut.is_cuda)
        coin = random.random()
        
        if coin < epsilon:
            action = random.randint(0, 1)
        else:
            action = qOut.argmax().item()
    
        return action