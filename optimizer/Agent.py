import torch.nn as nn
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
        for _ in range(10):
            s, a, r, sPrime = self.memory.sample()
            
            output = self.model(s)
            actualQ = output.gather(1, a.type(torch.int64))
            maxQPrime = self.model((sPrime)).max(1)[0].unsqueeze(1)
            targetQ = r + Config.gamma * maxQPrime
            
            # print(actualQ.shape, targetQ.shape)
            
            loss = self.lossFunction(targetQ, actualQ)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            weight_histograms(writer, index, self.model)
            
            totalLoss += loss
        writer.add_scalar("Loss", totalLoss, index)
        print(f'Loss: {totalLoss / 10}')
            
    def getAction(self, observation, epsilon):
        observation = torch.from_numpy(observation).type(torch.float).to(self.device).unsqueeze(0)
        qOut = self.model(observation)
        # print(qOut.is_cuda)
        coin = random.random()
        
        if coin < epsilon:
            action = random.randint(0, 1)
        else:
            action = qOut.argmax().item()
    
        return action