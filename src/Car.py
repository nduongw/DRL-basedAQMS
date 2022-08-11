import numpy as np
import random
from copy import copy
from config import Config
from src.Package import Package


class Car:
    coverRange = Config.coverRange
    observationRange = 2 * Config.coverRange
    
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.velocity = random.randint(Config.minVelocity, Config.maxVelocity)
        self.state = Config.action["OFF"]
        self.observation = np.zeros([2 * Car.observationRange + 1, 2 * Car.observationRange + 1])
        
    def setObservation(self, coverMap, carMap):
        copyCarMap = copy(carMap)
        copyCarMap[self.x, self.y] = 3
        paddedCoverMap = np.pad(coverMap, self.observationRange, constant_values=-1)
        paddedCarMap = np.pad(copyCarMap, self.observationRange, constant_values=-1)
        
        coverObservation = paddedCoverMap[self.x: self.x + 2 * self.observationRange + 1, self.y: self.y + 2 * self.observationRange + 1]
        carPosObservation = paddedCarMap[self.x: self.x + 2 * self.observationRange + 1, self.y: self.y + 2 * self.observationRange + 1]
        
        self.observation = np.stack([coverObservation, carPosObservation])
        
        
    def turnOn(self):
        self.state = Config.action["ON"]
    
    def turnOff(self):
        self.state = Config.action["OFF"]
        
    def run(self):
        self.x = self.x + self.velocity
    
    def action(self, server):
        print(f'Car {self.x} - {self.y} observation: ')
        print(self.observation[0])
        print(self.observation[1])
        print('\n')
        prob = abs(random.gauss(0, 1))
        
        if prob > 0.5:
            self.turnOn()
            print(f'Car at {self.x} - {self.y} turns on')
            package = Package(self.x, self.y)
            server.updateSentPackages(package)
            
        else:
            self.turnOff()
            print(f'Car at {self.x} - {self.y} turns off')