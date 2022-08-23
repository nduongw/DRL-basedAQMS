from ast import arg
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from tqdm import tqdm

from optimizer.Model import *
from optimizer.ReplayMemory import Memory
from optimizer.Agent import Agent
from config import Config

from src.Map import Map
from src.Server import GNBServer

parser = argparse.ArgumentParser()
parser.add_argument('--storepath', help='Location to store runs of tensorboard', required=True)
parser.add_argument('--model', help='Dense or CNN model', required=True)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f'runs/{args.storepath}')
# writer = SummaryWriter(f'runs/demo1')

modelList = {'DenseModel': DQNDenseModel(2, Config.obsShape, device),
             'CnnModel': DQNCNNModel(2, Config.obsShape, device)}

if args.model == 'dense':
    model = modelList['DenseModel'].to(device)
elif args.model == 'cnn':
    model = modelList['CnnModel'].to(device)
    
memory = Memory(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, optimizer, memory, device)

server = GNBServer()
map = Map(agent, server)
map.set_seed(42)

for i in tqdm(range(100000)):
    epsilon = max(0.01, 0.1 - 0.01 * (i / 200))
    map.run(epsilon, writer, memory, i)
    loss = 0
    '''
    for j in tqdm(range(100)):
        map.run(epsilon, writer, memory, j)
        # time.sleep(120)
    
    print(f'Memory size: {memory.size()}')
    writer.add_scalar('Reward', map.reward, i)
    
    map.resetMap()
    '''
    if memory.size() > Config.batchSize:
        loss = agent.train(i, writer)
    
    if i % 20 == 0 and i != 0:
        print(f'Step: {i}\tMemory size: {memory.size()}\tEpsilon : {epsilon: .2f}\tLoss: {loss: .5f}')

    writer.add_scalar('Epsilon', epsilon, i)
writer.close()