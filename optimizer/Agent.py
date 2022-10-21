import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import random
from config import Config
from utils import *

class Agent:
    def __init__(self, model, target_model, optimizer, memory, device) -> None:
        self.model = model
        self.target_model = target_model
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
            actualQ = output.gather(1, a.type(torch.int64))
            argMaxQPrime = self.model(sPrime).argmax(dim=1, keepdim=True)
            maxQPrime = self.target_model((sPrime)).gather(1, argMaxQPrime)
            
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
            
    def getAction(self, observation, epsilon, args):
        zoomCarMap = np.zeros([observation[0].shape[0] * args.zoom, observation[0].shape[0] * args.zoom])
        zoomCoverMap = np.zeros([observation[0].shape[0] * args.zoom, observation[0].shape[0] * args.zoom])
        
        for i in range(observation[0].shape[0]):
            for j in range(observation[0].shape[0]):
                if observation[0][i, j] != 0:
                    zoomCoverMap[args.zoom*i:args.zoom*i+args.zoom, args.zoom*j:args.zoom*j+args.zoom] = observation[0][i, j]
                if observation[1][i, j] != 0:
                    zoomCarMap[args.zoom*i:args.zoom*i+args.zoom, args.zoom*j:args.zoom*j+args.zoom] = observation[1][i, j]
        
        zoomObservation = np.stack([zoomCoverMap, zoomCarMap])
        zoomObservation = torch.from_numpy(zoomObservation)
        # print('Observation shape: ', zoomObservation.shape)
        zoomObservation = T.Resize((13, 13))(zoomObservation).type(torch.float).to(self.device).unsqueeze(0)
        # print('Observation after reshape: ', zoomObservation.shape)
        qOut = self.model(zoomObservation)
        # print(qOut.is_cuda)
        coin = random.random()
        
        if coin < epsilon:
            action = random.randint(0, 1)
        else:
            action = qOut.argmax().item()
    
        return action