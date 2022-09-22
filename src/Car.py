import numpy as np
import random
from copy import copy
from config import Config
from src.Package import Package

class Car:
    coverRange = Config.coverRange
    observationRange = 2 * Config.coverRange
    
    def __init__(self, x, y, agent) -> None:
        self.x = x
        self.y = y
        self.agent = agent
        self.velocity = 10
        self.state = Config.action["OFF"]
        self.observation = np.zeros([2, 2 * Car.observationRange + 1, 2 * Car.observationRange + 1])
        self.reward = 0
        self.nextObservation = np.zeros([2, 2 * Car.observationRange + 1, 2 * Car.observationRange + 1])
        
    def createObservation(self, coverMap, carMap):
        copyCarMap = copy(carMap)
        copyCarMap[self.x, self.y] = 106
        paddedCoverMap = np.pad(coverMap, self.observationRange, constant_values=192)
        paddedCarMap = np.pad(copyCarMap, self.observationRange, constant_values=192)
        
        coverObservation = paddedCoverMap[self.x: self.x + 2 * self.observationRange + 1, self.y: self.y + 2 * self.observationRange + 1]
        carPosObservation = paddedCarMap[self.x: self.x + 2 * self.observationRange + 1, self.y: self.y + 2 * self.observationRange + 1]
        
        return np.stack([coverObservation, carPosObservation])
    
    def setObservation(self, coverMap, carMap):
        self.observation = self.createObservation(coverMap, carMap)
        
    def setNextObservation(self, coverMap, carMap):
        self.nextObservation = self.createObservation(coverMap, carMap)
    
    def setReward(self, reward):
        self.reward = reward
        
    def turnOn(self):
        self.state = Config.action["ON"]
    
    def turnOff(self):
        self.state = Config.action["OFF"]
        
    def run(self, timeStep):
        # if timeStep % 300 < 100:
        #     self.x = self.x + int(self.velocity)
        # elif timeStep % 300  >= 100 and timeStep % 300 < 200:
        #     self.x = self.x + int(self.velocity * 2)
        # elif timeStep % 100  >= 200 and timeStep % 300 < 300:
        #     self.x = self.x + int(self.velocity / 2)
        self.x = self.x + self.velocity
            
    
    def action(self, server, epsilon, args):
        # * For random action:
        
        if not args.usingmodel:
            prob = abs(random.uniform(0, 1))
            
            if prob > 1 - args.sendingpercentage:
                self.turnOn()
                package = Package(self.x, self.y)
                server.updateSentPackages(package)
                
            else:
                self.turnOff()
        else:
            action = self.agent.getAction(self.observation, epsilon, args)
            
            if action == 1:
                self.turnOn()
                package = Package(self.x, self.y)
                server.updateSentPackages(package)
            else :
                self.turnOff()
 
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

if __name__ == '__main__':
    car = Car(98, 98, None)
    car.setObservation()
    