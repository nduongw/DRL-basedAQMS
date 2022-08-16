from ast import arg
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

from optimizer.Model import DQNModel
from optimizer.ReplayMemory import Memory
from optimizer.Agent import Agent
from config import Config

from src.Map import Map
from src.Server import GNBServer

parser = argparse.ArgumentParser()
parser.add_argument('--storepath', help='Location to store runs of tensorboard', required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f'runs/{args.storepath}')

model = DQNModel(2, Config.obsShape, device).to(device)
memory = Memory(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, optimizer, memory, device)

server = GNBServer()
map = Map(agent, server)
map.set_seed(42)

for i in range(15):
    epsilon = max(0.01, 0.08 - 0.01 * (i / 200))
    
    map.run(epsilon, writer, i)
    
    for car in map.carList:
        memory.add([car.observation / 255, car.state, car.reward, car.nextObservation])
    
    if memory.size() > 1000:
        agent.train(i, writer)
    

writer.close()