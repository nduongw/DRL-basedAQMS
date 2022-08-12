class Config:
    #MapParameters
    mapWidth = 50
    mapHeight = 100
    unCoverPeriod = 2
    
    state_size = 10
    action_size = 2
    minVelocity = 1
    maxVelocity = 3
    coverRange = 3
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 3
    decayRate = 0.2
    
    #OptimizerParameters
    bufferLimit = 100000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0005
    obsShape = [2, 13, 13]