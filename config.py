class Config_7:
    #MapParameters
    mapWidth = 200
    mapHeight = 800
    unCoverPeriod = 3
    
    state_size = 10
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 7
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    generationAmount = 20
    
    #OptimizerParameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0005
    obsShape = [2, 17, 17]
    carVelocity = 10
    
class Config2:
    #MapParameters
    mapWidth = 50
    mapHeight = 200
    unCoverPeriod = 10
    
    action_size = 2
    minVelocity = 2
    maxVelocity = 7
    coverRange = 3
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
    cLambda = 0.008

class Config3:
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
    cLambda = 0.008
    
    
class Config4:
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
    
class Config5:
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

class Config6:
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
    
class Config7:
    #Map Parameters
    mapWidth = 50
    mapHeight = 400
    unCoverPeriod = 5
    generationAmount = 10
    coverRange = 10
    
    #Car Parameters
    state_size = 10
    action_size = 2
    minVelocity = 1
    maxVelocity =20
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    #Optimizer Parameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 10
    cLambda = 0.005
    
class Config8:
    #Map Parameters
    mapWidth = 50
    mapHeight = 400
    unCoverPeriod = 5
    generationAmount = 10
    coverRange = 20
    
    #Car Parameters
    state_size = 10
    action_size = 2
    minVelocity = 1
    maxVelocity =20
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    #Optimizer Parameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 10
    cLambda = 0.01

class Config:
    #Map Parameters
    mapWidth = 50
    mapHeight = 400
    unCoverPeriod = 3
    generationAmount = 5
    coverRange = 5
    
    #Car Parameters
    state_size = 10
    action_size = 2
    minVelocity = 1
    maxVelocity = 1
    action = {
        "ON": 1,
        "OFF": 0
    }
    
    #Optimizer Parameters
    bufferLimit = 1500000
    batchSize = 32
    gamma = 0.98
    learningRate = 0.0025
    obsShape = [2, 13, 13]
    carVelocity = 10
    cLambda = 0.02