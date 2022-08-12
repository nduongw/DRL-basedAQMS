import torch.nn as nn
import torch
import random
from config import Config

class Agent:
    def __init__(self, model, optimizer, memory) -> None:
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
        self.lossFunction = nn.MSELoss()
            
    def train(self):
        totalLoss = 0
        for _ in range(10):
            s, a, r, sPrime = self.memory.sample()
            # print(s.shape)
            
            output = self.model(s)
            # print(output.shape)
            # print(a)
            actualQ = output.gather(1, a.type(torch.int64))
            maxQPrime = self.model((sPrime)).max(1)[0].unsqueeze(1)
            targetQ = r + Config.gamma * maxQPrime
            
            print(actualQ.shape, targetQ.shape)
            
            loss = self.lossFunction(targetQ, actualQ)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            totalLoss += loss
            print(f'Loss: {totalLoss / 10}')
            
    def getAction(self, observation, epsilon):
        qOut = self.model(observation)
        coin = random.random()
        
        if coin < epsilon:
            action = random.random(0, 1)
        else:
            action = qOut.argmax().item()
            
        return action