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

def calculateReward2(car, onRewardMap, offRewardMap, coverMap, previousCoverMap):
    '''
    If action is turn on -> Reward = (Total uncover area - Total cover area) / Cover range  
    Else                 -> Reward = (Total cover area - Total uncover area) / Cover range
    '''
    
    paddedPreviousMap = np.pad(previousCoverMap, 2 * Config.coverRange, constant_values=-1)
    paddedCoverMap = np.pad(coverMap, 2 * Config.coverRange, constant_values=-1)
    paddedOnRewardMap = np.pad(onRewardMap, 2 * Config.coverRange, constant_values=-1)
    paddedOffRewardMap = np.pad(offRewardMap, 2 * Config.coverRange, constant_values=-1)
    
    previousMap = paddedPreviousMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    presentMap = paddedCoverMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    onMap = paddedOnRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    offMap = paddedOffRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    
    reward = 0
    denominator = previousMap.shape[0] * previousMap.shape[1] - np.where(previousMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(previousMap == 0, 1, 0).sum() - np.where(previousMap == 1, 1, 0).sum()
        reward /= denominator
    else:
        reward = np.where(presentMap == 1, 1, 0).sum() - np.where(previousMap == 0, 1, 0).sum()
        reward /= denominator
    
    return reward

def calculateReward3(car, onRewardMap, offRewardMap, coverMap, previousCoverMap):
    '''
    If action is turn on -> Reward = (Total uncover area - Total cover area + Total uncover area of another cars - Total cover area of another cars) / Cover range  
    Else                 -> Reward = (Total cover area - Total uncover area - Total uncover area of another cars + Total cover area of another cars) / Cover range
    '''
    
    paddedPreviousMap = np.pad(previousCoverMap, 2 * Config.coverRange, constant_values=-1)
    paddedCoverMap = np.pad(coverMap, 2 * Config.coverRange, constant_values=-1)
    paddedOnRewardMap = np.pad(onRewardMap, 2 * Config.coverRange, constant_values=-1)
    paddedOffRewardMap = np.pad(offRewardMap, 2 * Config.coverRange, constant_values=-1)
    
    previousMap = paddedPreviousMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    presentMap = paddedCoverMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    onMap = paddedOnRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    offMap = paddedOffRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    
    reward = 0
    denominator = previousMap.shape[0] * previousMap.shape[1] - np.where(previousMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(previousMap == 0, 1, 0).sum() - np.where(previousMap == 1, 1, 0).sum() + \
                np.where(offMap > 1, offMap - 1, 0).sum() - np.where(onMap > 1, onMap - 1, 0).sum()
        reward /= (2 * denominator)
    else:
        reward = np.where(presentMap == 1, 1, 0).sum() - np.where(previousMap == 0, 1, 0).sum() - \
                np.where(offMap > 1, offMap - 1, 0).sum() + np.where(onMap > 1, onMap - 1, 0).sum()
        reward /= (2 * denominator)
    
    return reward

def calculateReward4(car, onRewardMap, offRewardMap, coverMap, previousCoverMap):
    '''
    If action is turn on -> Reward = (Total uncover area - Total cover area + Total uncover area of another cars - Total cover area of another cars) / Cover range  
    Else                 -> Reward = (Total cover area - Total uncover area - Total uncover area of another cars + Total cover area of another cars) / Cover range
    '''
    
    paddedPreviousMap = np.pad(previousCoverMap, 2 * Config.coverRange, constant_values=-1)
    paddedCoverMap = np.pad(coverMap, 2 * Config.coverRange, constant_values=-1)
    paddedOnRewardMap = np.pad(onRewardMap, 2 * Config.coverRange, constant_values=-1)
    paddedOffRewardMap = np.pad(offRewardMap, 2 * Config.coverRange, constant_values=-1)
    
    previousMap = paddedPreviousMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    presentMap = paddedCoverMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    onMap = paddedOnRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    offMap = paddedOffRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    
    reward = 0
    denominator = previousMap.shape[0] * previousMap.shape[1] - np.where(previousMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(previousMap == 0, 1, 0).sum() - np.where(previousMap == 1, 1, 0).sum() \
                - np.where(onMap > 1, onMap - 1, 0).sum()
        reward /= denominator
    else:
        reward = np.where(presentMap == 1, 1, 0).sum() - np.where(previousMap == 0, 1, 0).sum() - \
                np.where(offMap > 1, offMap - 1, 0).sum()
        reward /= denominator
    
    return reward
    
def calculateReward5(car, onRewardMap, offRewardMap, coverMap, previousCoverMap):
    '''
    If action is turn on -> Reward = (Total uncover area - Total cover area + Total uncover area of another cars - Total cover area of another cars) / Cover range  
    Else                 -> Reward = (Total cover area - Total uncover area - Total uncover area of another cars + Total cover area of another cars) / Cover range
    '''
    
    paddedPreviousMap = np.pad(previousCoverMap, 2 * Config.coverRange, constant_values=-1)
    paddedCoverMap = np.pad(coverMap, 2 * Config.coverRange, constant_values=-1)
    paddedOnRewardMap = np.pad(onRewardMap, 2 * Config.coverRange, constant_values=-1)
    paddedOffRewardMap = np.pad(offRewardMap, 2 * Config.coverRange, constant_values=-1)
    
    previousMap = paddedPreviousMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    presentMap = paddedCoverMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    onMap = paddedOnRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    offMap = paddedOffRewardMap[car.x + Config.coverRange: car.x + 3 * Config.coverRange + 1 , car.y + Config.coverRange: car.y + 3 * Config.coverRange + 1]
    
    reward = 0
    denominator = previousMap.shape[0] * previousMap.shape[1] - np.where(previousMap == -1, 1, 0).sum()
    
    if car.state == Config.action["ON"]:
        reward = np.where(previousMap == 0, 1, 0).sum() - np.where(previousMap == 1, 1, 0).sum() + \
                np.where(offMap > 1, offMap - 1, 0).sum()
        reward /= denominator
    else:
        reward = np.where(presentMap == 1, 1, 0).sum() - np.where(previousMap == 0, 1, 0).sum() \
                + np.where(onMap > 1, onMap - 1, 0).sum()
        reward /= (2 * denominator)
    
    return reward    
    
if __name__ == "__main__":
    car = Car(22, 0, None)
    rewardMap = np.zeros(50, 50)
    print(rewardMap)