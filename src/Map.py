from distutils.command.config import config
import numpy as np
import random
import scipy.stats

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
        self.server = server
        self.agent = agent
        self.time = 0
        self.reward = 0
        self.args = args
        self.coverRate = 0
        self.sendingRate = 0
        
    def run(self, epsilon, writer, memory, step, isTest=False, testStep=0, csvWriter=None):
        onRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        offRewardMap = np.zeros([self.mapHeight, self.mapWidth])
        totalReward = 0
        if len(self.carList) > 0:
            for car in self.carList:
                car.setObservation(self.coverMap, self.carPosMap)
                if self.args.pso:
                    if step % self.unCoverPeriod == 0:
                        car.action(self.server, epsilon, self.args)
                        if car.state == Config.action["ON"]:
                            self.carPosMap[car.x, car.y] = 222
                        else:
                            self.carPosMap[car.x, car.y] = 1
                    else:
                        car.state = Config.action["OFF"]
                else:
                    car.action(self.server, epsilon, self.args)
                    if car.state == Config.action["ON"]:
                        self.carPosMap[car.x, car.y] = 222
                    else:
                        self.carPosMap[car.x, car.y] = 1
                    
                previousCoverMap = np.copy(self.coverMap)
                self.updateCoverMap(car, onRewardMap, offRewardMap)
                self.coverMap = np.where(onRewardMap >= 1, 1, self.coverMap)
                car.setNextObservation(self.coverMap, self.carPosMap)
                self.calcReward(car, previousCoverMap, onRewardMap, offRewardMap)
                #Write to log file
                if csvWriter:
                    xStart = max(0, car.x - Car.coverRange)
                    xEnd = min(self.mapHeight, car.x + Car.coverRange + 1)
                    yStart = max(0, car.y - Car.coverRange)
                    yEnd = min(self.mapWidth, car.y + Car.coverRange + 1)
                    
                    carCoverMap = previousCoverMap[xStart: xEnd, yStart: yEnd]
                    carCoverMapPrime = self.coverMap[xStart: xEnd, yStart: yEnd]
    
                    csvWriter.writerow([car.x, car.y, np.where(carCoverMap == 0, 1, 0).sum(), car.state, np.where(carCoverMapPrime == 0, 1, 0).sum(), car.reward])
                totalReward += car.reward
                
            self.reward += totalReward
            
            if isTest == False:
                for car in self.carList:
                    # car.setNextObservation(self.coverMap, self.carPosMap)
                    memory.add([car.observation, car.state, car.reward, car.nextObservation])
            
            # for car in self.carList:
            #     writer.add_image(f'Car {car.x}-{car.y} observation at {step}', car.observation[0], 0, dataformats='HW')
            #     writer.add_image(f'Car {car.x}-{car.y} next observation at {step}', car.nextObservation[0], 0, dataformats='HW')
            
            # if step % 100 == 0 and step > 0:
            #     sampleList = random.sample(self.carList, 5)
            #     for car in sampleList:
            #         writer.add_image(f'[Test 100] Cover map of car{car.x}-{car.y} at step:{step}', car.observation[0], 0, dataformats='HW')
            #         writer.add_image(f'[Test 100] Car position map of car{car.x}-{car.y} at step:{step}', car.observation[1], 0, dataformats='HW')
            #         writer.add_image(f'[Test 100] Next observation map of car{car.x}-{car.y} at step:{step}', car.nextObservation[0], 0, dataformats='HW')

            if testStep != 0 and isTest == True:
                if self.args.pso:
                    if step % self.unCoverPeriod == 0:
                        writer.add_scalar(f'Cover rate at test step {testStep}', self.calcCoverRate(), step)
                        writer.add_scalar(f'Overlap rateat test step {testStep}', self.calcOverlapRate(previousCoverMap, onRewardMap), step)
                        writer.add_scalar(f'Car overlap rate at test step {testStep}', self.calcCarOverlap(onRewardMap), step)
                        writer.add_scalar(f'Number of car at test step {testStep}', len(self.carList), step)
                        writer.add_scalar(f'Sending rate at test step {testStep}', self.countOnCar() / len(self.carList), step)
                        self.coverRate += self.calcCoverRate()
                        self.sendingRate += self.countOnCar() / len(self.carList)
                else:
                    writer.add_scalar(f'Cover rate at test step {testStep}', self.calcCoverRate(), step)
                    writer.add_scalar(f'Overlap rateat test step {testStep}', self.calcOverlapRate(previousCoverMap, onRewardMap), step)
                    writer.add_scalar(f'Car overlap rate at test step {testStep}', self.calcCarOverlap(onRewardMap), step)
                    writer.add_scalar(f'Number of car at test step {testStep}', len(self.carList), step)
                    writer.add_scalar(f'Sending rate at test step {testStep}', self.countOnCar() / len(self.carList), step)
                    self.coverRate += self.calcCoverRate()
                    self.sendingRate += self.countOnCar() / len(self.carList)
                
            for car in self.carList:
                car.run(step)
        
        if self.time % self.unCoverPeriod == 0 and self.time != 0:
            self.coverMap -= 1
            self.coverMap = np.where(self.coverMap > 0, self.coverMap, 0)

        self.removeInvalidCar()
        self.generateCar(step)
        self.updateCarPosition()

        self.time += 1
        
    def generateCar(self, timeStep):
        addedCarAmount = self.generatePoissonDistribution(Config.cLambda, timeStep)
        
        for _ in range(addedCarAmount):
            addedCar = Car(random.randint(0, Config.maxVelocity), random.randint(0, self.mapWidth - 1), self.agent)
            addedCar.setObservation(self.coverMap, self.carPosMap)
            self.carList.append(addedCar)
    
    def generatePoissonDistribution(self, clambda, timeStep):
        addedCarAmount = -1
        currentCarAmount = np.where(self.carPosMap != 0, 1, 0).sum()
        if timeStep == 0:
            while addedCarAmount > (self.mapWidth * Config.carVelocity) or addedCarAmount < 0:
                addedCarAmount = scipy.stats.poisson(self.mapHeight * self.mapWidth * clambda).rvs()

            return addedCarAmount
        
        while addedCarAmount - currentCarAmount < 0 or addedCarAmount < 0:
            addedCarAmount = scipy.stats.poisson(self.mapHeight * self.mapWidth * clambda).rvs()

        return addedCarAmount - currentCarAmount

    def generateFixedDistribution(self):
        addedCarAmount = random.randint(0, Config.generationAmount)
        
        return addedCarAmount
        
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
        
    def updateCoverMap(self, car, onRewardMap, offRewardMap):
        if car.state == Config.action["ON"]:
            self.setOnCover(car.x, car.y, onRewardMap)
        else:
            self.setOffCover(car.x, car.y, offRewardMap)
        
    def setOnCover(self, x, y, onRewardMap):
        for i in range(max(0, x - Car.coverRange), min(self.mapHeight, x + Car.coverRange + 1)):
            for j in range(max(0, y - Car.coverRange), min(self.mapWidth, y + Car.coverRange + 1)):
                onRewardMap[i, j] += 1
    
    def setOffCover(self, x, y, offRewardMap):
        for i in range(max(0, x - Car.coverRange), min(self.mapHeight, x + Car.coverRange + 1)):
            for j in range(max(0, y - Car.coverRange), min(self.mapWidth, y + Car.coverRange + 1)):
                offRewardMap[i, j] += 1
                
    def addCar(self, car):
        self.carList.append(car)
        
    def calcReward(self, car, previousCoverMap, onRewardMap, offRewardMap):
        if self.args.rewardfunc == 'ver1':
            reward = calculateReward(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver2':
            reward = calculateReward2(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver3':
            reward = calculateReward3(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver4':
            reward = calculateReward4(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver5':
            reward = calculateReward5(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver6':
            reward = calculateReward6(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap)
        elif self.args.rewardfunc == 'ver7':
            reward = calculateReward7(car, onRewardMap, offRewardMap, self.coverMap, previousCoverMap, self.carPosMap)
            
        car.setReward(reward)
    
    def calcCoverRate(self):
        coverRate = self.coverMap.sum() / (self.mapHeight * self.mapWidth)
        return coverRate
    
    def countOnCar(self):
        count = 0
        for car in self.carList:
            if car.state == Config.action["ON"]:
                count += 1
        return count
    
    def calcOverlapRate(self, previousCoverMap, onRewardMap):
        overlap = np.where(previousCoverMap * onRewardMap > 0 , 1, 0).sum()
        overlap /= (Config.mapHeight * Config.mapWidth)
        return overlap

    def calcCarOverlap(self, onRewardMap):
        overlapMap = onRewardMap - 1
        overlap = np.where(overlapMap > 0, 1, 0).sum()
        if self.countOnCar() != 0:
            overlap /= (self.countOnCar() * (Config.coverRange * 2 + 1))
        else: 
            overlap = 0        
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
        self.time = 0
        self.server.resetServer()
        self.reward = 0
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        for car in self.carList:
            car.set_seed(seed)