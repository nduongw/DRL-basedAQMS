import numpy as np
import random
from PIL import Image

from config import Config
from src.Car import Car
from optimizer.RewardFunction import calculateReward

class Map:
    def __init__(self, agent, server) -> None:
        self.mapWidth = Config.mapWidth
        self.mapHeight = Config.mapHeight
        self.carList = []
        self.coverMap = np.zeros([self.mapHeight, self.mapWidth])
        self.carPosMap = np.zeros([self.mapHeight, self.mapWidth])
        self.unCoverPeriod = Config.unCoverPeriod
        self.rewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.server = server
        self.agent = agent
        
    def run(self, epsilon):
        print('Car map:')
        print(self.carPosMap)
        print('\n')
        print('Cover map before cars do action:')
        print(self.coverMap)
        print('\n')
        for car in self.carList:
            car.setObservation(self.coverMap, self.carPosMap)
        
        for car in self.carList:
            car.action(self.server, epsilon)
        
        self.updateCoverMap()
        self.calcReward()
        
        for _ in range(self.unCoverPeriod):
            self.coverMap -= Config.decayRate
            self.coverMap = np.where(self.coverMap > 0, self.coverMap, 0)
            
            for car in self.carList:
                car.run()
            
            self.generateCar()
            self.removeInvalidCar()
            self.updateCarPosition()
            print('Cover map after cars do action: ')
            print(self.coverMap)
            print('\n')
            
        print(f'Total sent packages: {self.server.getTotalPackages()}')
        
        for car in self.carList:
            car.setNextObservation(self.coverMap, self.carPosMap)
        
    def generateCar(self):
        addedCarAmount = random.randint(0, Config.generationAmount)
        
        for _ in range(addedCarAmount):
            addedCar = Car(random.randint(0, Config.maxVelocity), random.randint(0, self.mapWidth - 1), self.agent)
            addedCar.setObservation(self.coverMap, self.carPosMap)
            self.carList.append(addedCar)
        
    def removeInvalidCar(self):
        removeCarList = []
        for car in self.carList:
            if car.x >= self.mapHeight:
                removeCarList.append(car)
        
        for removedCar in removeCarList:
            for car in self.carList:
                if removedCar == car:
                    self.carList.remove(car)
            
    def updateCarPosition(self):
        self.carPosMap = np.zeros([Config.mapHeight, Config.mapWidth])
        for car in self.carList:
            self.carPosMap[car.x, car.y] = 1
        print('Car position map')
        print(self.carPosMap)
        
    def updateCoverMap(self):
        for car in self.carList:
            if car.state == Config.action["ON"]:
                self.setCover(car.x, car.y)
        
        self.coverMap = np.where(self.rewardMap > 1, 1, self.coverMap)
        print('Reward map')
        print(self.rewardMap)
        print('Cover map')
        print(self.coverMap)
        
    def setCover(self, x, y):
        for i in range(max(0, x - Car.coverRange), min(self.mapHeight, x + Car.coverRange + 1)):
            for j in range(max(0, y - Car.coverRange), min(self.mapWidth, y + Car.coverRange + 1)):
                self.rewardMap[i, j] += 1
        self.coverMap = np.where(self.rewardMap >= 1, 1, self.coverMap)
                
                
    def addCar(self, car):
        self.carList.append(car)
        
    def calcReward(self):
        for car in self.carList:
            calculateReward(car)
    
    def showCarMap(self, time):
        # simg = np.stack((self.carPosMap, self.carPosMap, self.carPosMap), axis=0)
        # print(simg.shape)
        # img = Image.fromarray(simg, 'L')
        # img.save(f'CarPosition{time}.png')
        # coverimg = Image.fromarray(self.coverMap, 'L')
        # coverimg.save(f'CoverMap{time}.png')
        # # img.show()
        print('Car\'s position map')
        print(self.carPosMap)
        print('Cover map')
        print(self.coverMap)
        # print(coverimg)