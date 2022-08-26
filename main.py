import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from optimizer.Model import *
from optimizer.ReplayMemory import Memory
from optimizer.Agent import Agent
from config import Config

from src.Map import Map
from src.Server import GNBServer

if not os.path.exists('models'):
    os.mkdir('models')
        
if not os.path.exists('runs'):
    os.mkdir('runs')

parser = argparse.ArgumentParser()
parser.add_argument('--storepath',type=str, help='Location to store runs of tensorboard', required=True)
parser.add_argument('--model',type=str, help='Dense or CNN model', required=True)
parser.add_argument('--modelpath',type=str, help='Name of saved model state dict', required=False)
parser.add_argument('--rewardfunc',type=str, help='Rerward version which you want to use', required=False)
args = parser.parse_args()

#seed for model parameters
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f'runs/{args.storepath}')
# writer = SummaryWriter(f'runs/demo1')

modelList = {'DenseModel': DQNDenseModel(2, Config.obsShape, device),
             'CnnModel': DQNCNNModel(2, Config.obsShape, device)}

if args.model == 'dense':
    model = modelList['DenseModel'].to(device)
    target_model = modelList['DenseModel'].to(device)
elif args.model == 'cnn':
    model = modelList['CnnModel'].to(device)    
    target_model = modelList['CnnModel'].to(device)
    
target_model.load_state_dict(model.state_dict())

# for testing model
# model.load_state_dict(torch.load('model/dense-238d17h21-atStep6000.pt'))

memory = Memory(device)
optimizer = optim.Adam(model.parameters(), lr=Config.learningRate)
agent = Agent(model, target_model, optimizer, memory, device)

server = GNBServer()
map = Map(agent, server, args)
testMap = Map(agent, server, args)
map.set_seed(42)
testMap.set_seed(42)
testStep = 1

def testModel(testMap, testStep, step):
    print(f'Testing phase {step}:\n')
    testMap.resetMap()
    epsilon = 0
    for i in tqdm(range(1000)):
        testMap.run(epsilon, writer, memory, i, isTest=True, testStep=testStep)
    
    writer.add_scalar('Reward', testMap.reward / 1000, testStep)
    print(f'Reward of testing phase; {testMap.reward / 1000}')
    testStep += 1
    return testMap.reward / 1000
        
if __name__ == "__main__":
    # '''
    for i in tqdm(range(50000)):
        bestReward = -9999
        loss = 0
        epsilon = max(0.01, 0.1 - 0.01 * (i / 200))
        writer.add_scalar('Epsilon', epsilon, i)
        
        #simulation phase
        map.run(epsilon, writer, memory, i, isTest=False, testStep=testStep)

        #training phase
        if memory.size() > Config.batchSize:
            loss = agent.train(i, writer)
        
        if i % 20 == 0 and i != 0:
            print(f'Step: {i}\tMemory size: {memory.size()}\tEpsilon : {epsilon: .2f}\tLoss: {loss: .5f}')
            loss = 0

        #save model
        if i % 50 == 0 and i != 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        #testing phase
        if i % 100 == 0:
            reward = testModel(testMap, testStep, i)
            if reward >= bestReward:
                print(f'Reward increased from {bestReward} to {reward} => saving model...\n')
                torch.save(model.state_dict(), f'models/{args.model}-{args.modelpath}-bestRewardAtStep{i}.pt')
                bestReward = reward
            testStep += 1
    # '''
    # testModel(testMap, testStep, 1)        
    
    writer.close()