import numpy as np
import random

from config import Config
from src.Car import Car
from optimizer.RewardFunction import *

class Map:
    def __init__(self, agent, server, args) -> None:
        self.mapWidth = Config.mapWidth
        self.mapHeight = Config.mapHeight
        self.carList = []
        self.coverMap = np.zeros([self.mapHeight, self.mapWidth])
        self.carPosMap = np.zeros([self.mapHeight, self.mapWidth])
        self.unCoverPeriod = Config.unCoverPeriod
        self.onRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.offRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.server = server
        self.agent = agent
        self.time = 0
        self.reward = 0
        self.args = args
        
    def run(self, epsilon, writer, memory, step, isTest=False, testStep=0):
        self.resetRewardMap()
        if len(self.carList) > 0:
            for car in self.carList:
                car.setObservation(self.coverMap, self.carPosMap)
            
            for car in self.carList:
                car.action(self.server, epsilon)
            
            previousCoverMap = np.copy(self.coverMap)
            self.updateCoverMap()
            
            totalReward = self.calcReward(previousCoverMap)
            self.reward += totalReward
            
            if isTest == False:
                for car in self.carList:
                    if car.state == Config.action["ON"]:
                        self.carPosMap[car.x, car.y] = 2
                    else:
                        self.carPosMap[car.x, car.y] = 1
                    car.setNextObservation(self.coverMap, self.carPosMap)
                    memory.add([car.observation, car.state, car.reward, car.nextObservation])
            
            # for car in self.carList:
            #     writer.add_image(f'Car {car.x}-{car.y} observation at {step}', car.observation[0], 0, dataformats='HW')
            #     writer.add_image(f'Car {car.x}-{car.y} next observation at {step}', car.nextObservation[0], 0, dataformats='HW')
            
            # if step % 100 == 0 and step > 0:
            #     sampleList = random.sample(self.carList, 5)
            #     for car in sampleList:
            #         writer.add_image(f'Cover map of car{car.x}-{car.y} at step:{step}', car.observation[0], 0, dataformats='HW')
            #         writer.add_image(f'Car position map of car{car.x}-{car.y} at step:{step}', car.observation[1], 0, dataformats='HW')
            #         writer.add_image(f'Next observation map of car{car.x}-{car.y} at step:{step}', car.nextObservation[0], 0, dataformats='HW')

            if testStep != 0 and isTest == True:
                writer.add_scalar(f'Cover rate at test step {testStep}', self.calcCoverRate(), step)
                writer.add_scalar(f'Overlap rateat test step {testStep}', self.calcOverlapRate(previousCoverMap), step)
                writer.add_scalar(f'Car overlap rate at test step {testStep}', self.calcCarOverlap(), step)
                writer.add_scalar(f'Number of car at test step {testStep}', len(self.carList), step)
                writer.add_scalar(f'Sending rate at test step {testStep}', self.countOnCar() / len(self.carList), step)
            
            for car in self.carList:
                car.run()
        
        if self.time % self.unCoverPeriod == 0 and self.time != 0:
            self.coverMap -= 1
            self.coverMap = np.where(self.coverMap > 0, self.coverMap, 0)

        self.generateCar()
        self.removeInvalidCar()
        self.updateCarPosition()

        self.time += 1
        
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
        
    def updateCoverMap(self):
        for car in self.carList:
            if car.state == Config.action["ON"]:
                self.setOnCover(car.x, car.y)
            else:
                self.setOffCover(car.x, car.y)
        self.coverMap = np.where(self.onRewardMap >= 1, 1, self.coverMap)
        
    def setOnCover(self, x, y):
        for i in range(max(0, x - Car.coverRange), min(self.mapHeight, x + Car.coverRange + 1)):
            for j in range(max(0, y - Car.coverRange), min(self.mapWidth, y + Car.coverRange + 1)):
                self.onRewardMap[i, j] += 1
    
    def setOffCover(self, x, y):
        for i in range(max(0, x - Car.coverRange), min(self.mapHeight, x + Car.coverRange + 1)):
            for j in range(max(0, y - Car.coverRange), min(self.mapWidth, y + Car.coverRange + 1)):
                self.offRewardMap[i, j] += 1
                
    def addCar(self, car):
        self.carList.append(car)
        
    def calcReward(self, previousCoverMap):
        totalReward = 0
        for car in self.carList:
            if self.args.rewardfunc == 'ver1':
                reward = calculateReward(car, self.onRewardMap, self.offRewardMap, self.coverMap, previousCoverMap)
            elif self.args.rewardfunc == 'ver2':
                reward = calculateReward2(car, self.onRewardMap, self.offRewardMap, self.coverMap, previousCoverMap)
            elif self.args.rewardfunc == 'ver3':
                reward = calculateReward3(car, self.onRewardMap, self.offRewardMap, self.coverMap, previousCoverMap)
            elif self.args.rewardfunc == 'ver4':
                reward = calculateReward4(car, self.onRewardMap, self.offRewardMap, self.coverMap, previousCoverMap)
            elif self.args.rewardfunc == 'ver5':
                reward = calculateReward5(car, self.onRewardMap, self.offRewardMap, self.coverMap, previousCoverMap)
            
            car.setReward(reward)
            totalReward += reward

        # totalReward /= len(self.carList)
        return totalReward

    def resetRewardMap(self):
        self.onRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.offRewardMap = np.zeros([self.mapHeight, self.mapWidth])
    
    def calcCoverRate(self):
        coverRate = self.coverMap.sum() / (self.mapHeight * self.mapWidth)
        return coverRate
    
    def countOnCar(self):
        count = 0
        for car in self.carList:
            if car.state == Config.action["ON"]:
                count += 1
        return count
    
    def calcOverlapRate(self, previousCoverMap):
        overlap = np.where(previousCoverMap * self.onRewardMap > 0 , 1, 0).sum()
        overlap /= (Config.mapHeight * Config.mapWidth)
        return overlap

    def calcCarOverlap(self):
        overlapMap = self.onRewardMap - 1
        overlap = np.where(overlapMap > 0, overlapMap, 0).sum()
        overlap /= (self.countOnCar() * (Config.coverRange * 2 + 1))        
        return overlap
        
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
    
    def resetMap(self):
        self.carList = []
        self.coverMap = np.zeros([self.mapHeight, self.mapWidth])
        self.carPosMap = np.zeros([self.mapHeight, self.mapWidth])
        self.onRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.offRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        self.time = 0
        self.server.resetServer()
        self.reward = 0
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        for car in self.carList:
            car.set_seed(seed)