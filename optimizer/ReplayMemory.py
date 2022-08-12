from collections import deque
import random
import torch

from config import Config

class Memory:
    def __init__(self) -> None:
        self.buffer = deque(maxlen=Config.bufferLimit)
        
    def add(self, transition):
        self.buffer.append(transition)
        
    def sample(self):
        miniBatch = random.sample(self.buffer, Config.batchSize)
        sLst, aLst, rLst, sPrimeLst = [], [], [], []
        
        for transition in miniBatch:
            s, a, r, sPrime = transition
            sLst.append(s)
            aLst.append([a])
            rLst.append([r])
            sPrimeLst.append([sPrime])
            
        return torch.tensor(sLst, dtype=torch.float), \
                torch.tensor(aLst, dtype=torch.float), \
                torch.tensor(rLst, dtype=torch.float), \
                torch.tensor(sPrimeLst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)
                