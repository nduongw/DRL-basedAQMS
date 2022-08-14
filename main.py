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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DQNModel(2, Config.obsShape, device).to(device)
memory = Memory(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, optimizer, memory, device)

server = GNBServer()
map = Map(agent, server)


for i in range(10000):
    # print(f'\nRun {i}')
    epsilon = max(0.01, 0.08 - 0.01 * (i / 200))
    
    map.run(epsilon)
    
    for car in map.carList:
        memory.add([car.observation / 10, car.state, car.reward, car.nextObservation])
    
    if memory.size() > 3000:
        agent.train()