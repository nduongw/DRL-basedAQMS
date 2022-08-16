from turtle import onrelease
import numpy as np
from config import Config
from src.Car import Car

def calculateReward(car, rewardMap):
    # print('Car action', car.state)
    paddedRewardMap = np.pad(rewardMap, 2 * Config.coverRange, constant_values=-1)
    # print(paddedRewardMap.shape)
    rewardCarMap = paddedRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    # print(rewardCarMap.shape)
    # print(rewardCarMap)
    reward = 0
    denominator = rewardCarMap.shape[0] * rewardCarMap.shape[1] - np.where(rewardCarMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(rewardCarMap == 1, 1, 0).sum() - np.where(rewardCarMap > 1, rewardCarMap - 1, 0).sum()
        reward /= denominator    
    else:
        reward = np.where(rewardCarMap > 0, 1, 0).sum() - np.where(rewardCarMap == 0, 1, 0).sum()
        reward /= denominator
        
    return reward

if __name__ == "__main__":
    car = Car(22, 0, None)
    rewardMap = np.zeros(50, 50)
    print(rewardMap)