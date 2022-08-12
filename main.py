from xmlrpc.client import Server
import torch
import torch.optim as optim
import torch.nn.functional as F

from optimizer.Model import DQNModel
from optimizer.ReplayMemory import Memory
from optimizer.Agent import Agent
from config import Config

from src.Map import Map
from src.Server import GNBServer

model = DQNModel(2, Config.obsShape)
memory = Memory()
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, optimizer, memory)

server = GNBServer()
map = Map(agent, server)

for i in range(10):
    epsilon = max(0.01, 0.08 - 0.01 * (i / 200))
    
    map.run(epsilon)
    
    # for car in map.carList:
    #     memory.add([car.observation, car.state, car.reward, car.nextObservation])
    
    # if memory.size > 3000:
    #     agent.train()