class Config:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 8
    
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
    
    #logs
    storePath = 'runs/x90'