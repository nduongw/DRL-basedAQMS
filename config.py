class Config:
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
    
    #OptimizerParameters
    bufferLimit = 100000
    batchSize = 32