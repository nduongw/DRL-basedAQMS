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
# writer = SummaryWriter(f'runs/demo1')

model = DQNModel(2, Config.obsShape, device).to(device)
memory = Memory(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, optimizer, memory, device)

server = GNBServer()
map = Map(agent, server)
map.set_seed(42)

for i in range(100000):
    epsilon = max(0.01, 0.1 - 0.01 * (i / 400))
    
    map.run(epsilon, writer, memory, i)
    
    if memory.size() > 3000:
        agent.train(i, writer)
    
    if i % 50 == 0 and i != 0:
        print(f'Step: {i}: Memory size: {memory.size()} - Epsilon : {epsilon}')

writer.close()