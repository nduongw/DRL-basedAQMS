from collections import deque
import random
import torch
import numpy as np

from config import Config

class Memory:
    def __init__(self, device) -> None:
        self.buffer = deque(maxlen=Config.bufferLimit)
        self.device = device
        
    def add(self, transition):
        self.buffer.append(transition)
        
    def sample(self):
        miniBatch = random.sample(self.buffer, Config.batchSize)
        sLst, aLst, rLst, sPrimeLst = [], [], [], []
        
        for transition in miniBatch:
            s, a, r, sPrime = transition
            sLst.append(s / 255.0)
            aLst.append([a])
            rLst.append([r])
            sPrimeLst.append(sPrime / 255.0)
            
        return torch.tensor(np.array(sLst), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(aLst), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(rLst), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(sPrimeLst), dtype=torch.float).to(self.device)
    
    def size(self):
        return len(self.buffer)
                