import numpy as np
import random

from stuff import objectiveFunction
from config import Config

class PSOBased:
    def __init__(self, outputDim, xMin, xMax, vMin, vMax, nbParticle, gBestValue, loopTimes, L1, epsilon, c1, wInit):
        self.outputDim = outputDim
        self.xMin = xMin
        self.xMax = xMax
        self.vMin = vMin
        self.vMax = vMax
        self.nbParticle = nbParticle
        self.gBest = np.zeros((self.outputDim))
        self.gBestValue = gBestValue
        self.loopTimes = loopTimes
        self.L1 = L1
        self.epsilon = epsilon
        self.c1 = c1
        self.wInit = wInit
        
    def optimizer(self):
        self.gBestValue = 0
        x = np.random.rand(self.nbParticle, self.outputDim)
        v = np.zeros((self.nbParticle, self.outputDim))
        #khoi tao quan the ngau nhien
        for i in range(self.nbParticle):
            if objectiveFunction(x[i], Config) > self.gBestValue:
                self.gBestValue = objectiveFunction(x[i], Config)
                self.gBest = x[i]
        
        for nbIterations in range(0, self.loopTimes):
            w = self.wInit * (self.loopTimes - 0.8 * nbIterations) / self.loopTimes
            
            for i in range(self.nbParticle):
                if nbIterations % self.L1 == 0:
                    if (v[i]**2).sum() < self.epsilon:
                        randomVelocity = random.random() * (self.vMax - self.vMin) + self.vMin
                        nbDims = np.random.randint(1, self.outputDim + 1)
                        nbDimsList = np.random.choice(range(self.outputDim), nbDims, replace=False)
                        for dim in nbDimsList:
                            v[i][dim] = randomVelocity
                            
                for j in range(0, self.outputDim):
                    v[i][j] = w * v[i][j] + self.c1 * random.random() * (self.gBest[j] - x[i][j])
                    v[i][j] = max(min(v[i][j], self.vMax), self.vMin)
                    x[i][j] = x[i][j] + v[i][j]
                    x[i][j] = max(min(x[i][j], self.xMax), self.xMin)
            
            if objectiveFunction(x[i], Config) > self.gBestValue:
                self.gBestValue = objectiveFunction(x[i], Config)
                self.gBest = x[i]
        
        print(f'gBest: {self.gBest}')
        print(f'gBest value: {self.gBestValue}')
        return self.gBest        
        
if __name__ == "__main__":
    pso = PSOBased(2, 0, 1, 0.2, -0.2, 10, 0, 1000, 10, 1e-8, 0, 0.8)
    print(pso.optimizer())