class Config:
    #MapParameters
    mapWidth = 10
    mapHeight = 10
    unCoverPeriod = 5
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 5
    coverRange = 1
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 10
    decayRate = 0.2
    
    #OptimizerParameters
    bufferLimit = 100000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0005
    obsShape = [2, 13, 13]