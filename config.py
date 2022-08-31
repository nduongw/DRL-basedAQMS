class Config:
    #MapParameters
    mapWidth = 100
    mapHeight = 400
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
    
    generationAmount = 5
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    
    #logs
    storePath = 'runs/x90'