import torch.nn as nn
import random
from config import Config

class Agent:
    def __init__(self, model, optimizer, memory) -> None:
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
            
    def train(self):
        for _ in range(10):
            s, a, r, sPrime = self.memory.sample()
            
            output = self.model(s)
            actualQ = output.gather(1, a)
            maxQPrime = self.model((sPrime)).max(1)[0].unsqueeze(1)
            targetQ = r + Config.gamma * maxQPrime
            
            loss = nn.MSELoss(targetQ, actualQ)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def getAction(self, observation, epsilon):
        qOut = self.model(observation)
        coin = random.random()
        
        if coin < epsilon:
            action = random.random(0, 1)
        else:
            action = qOut.argmax().item()
            
        return action