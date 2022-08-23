class Config:
    #MapParameters
    mapWidth = 100
    mapHeight = 200
    unCoverPeriod = 6
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 4
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 10
    
    #OptimizerParameters
    bufferLimit = 5000000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.00025
    obsShape = [2, 13, 13]
    
    #logs
    storePath = 'runs/x90'