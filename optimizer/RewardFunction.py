from turtle import onrelease
import numpy as np
from config import Config

def calculateReward(car, rewardMap):
    paddedRewardMap = np.pad(rewardMap, 2 * Config.coverRange, constant_values=-1)
    rewardCarMap = paddedRewardMap[car.x: car.x + 2 * 2 * Config.coverRange + 1, car.y: car.y + 2 * 2 * Config.coverRange + 1]
    reward = 0
    denominator = rewardCarMap.shape[0] * rewardCarMap.shape[1] - np.where(rewardCarMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(rewardCarMap == 1, 1, 0).sum() - np.where(rewardCarMap > 1, rewardCarMap, 0).sum()
        reward /= denominator    
    else:
        reward = np.where(rewardCarMap != 0, 1, 0).sum() - np.where(rewardCarMap == 0, 1, 0).sum()
        reward /= denominator
        
    return reward