from collections import deque
import random
import torch
import numpy as np
import torchvision.transforms as T

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
            s_t = T.Resize((17, 17))(torch.from_numpy(s)).numpy()
            sPrime_t = T.Resize((17, 17))(torch.from_numpy(sPrime)).numpy()
            
            sLst.append(s_t / 255.0)
            aLst.append([a])
            rLst.append([np.clip(r, -1, 1)])
            sPrimeLst.append(sPrime_t / 255.0)
            
            sTensor = torch.tensor(np.array(sLst), dtype=torch.float).to(self.device)
            aTensor = torch.tensor(np.array(aLst), dtype=torch.float).to(self.device)
            rTensor = torch.tensor(np.array(rLst), dtype=torch.float).to(self.device)
            sPrimeTensor = torch.tensor(np.array(sPrimeLst), dtype=torch.float).to(self.device)
            
        return sTensor, aTensor, rTensor, sPrimeTensor
        
    def size(self):
        return len(self.buffer)
                