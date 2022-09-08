class Config:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 3
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 5
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 5
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 17, 17]
    carVelocity = 10
    
class Config_2:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 10
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 10
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 7
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 10

class Config_3:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 20
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 8
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 4
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 5
    
class Config_4:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 20
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 8
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 8
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 1
    
class Config_5:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 2
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 5
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 8
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 10

class Config_6:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 2
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 5
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 20
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 17, 17]
    carVelocity = 10
    
    velocityRange = [1, 2, 10]
    velocityRange2 = [1, 2, 5]
    velocityRange3 = [1, 2, 2]
    velocityRange4 = [1, 2, 0.5]
    velocityRange5 = [0.5, 2, 0.5]
    

