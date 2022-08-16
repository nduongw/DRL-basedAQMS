class Config:
    #MapParameters
    mapWidth = 50
    mapHeight = 100
    unCoverPeriod = 5
    
    state_size = 10
    action_size = 2
    minVelocity = 1
    maxVelocity = 5
    coverRange = 3
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 5
    
    #OptimizerParameters
    bufferLimit = 1000000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0005
    obsShape = [2, 13, 13]
    
    #logs
    storePath = 'runs/x90'