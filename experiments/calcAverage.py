import pandas as pd

df = pd.read_csv('experiments/testing-config1-pso-poisson-changev/coverRate.csv')
print(df.mean())

df = pd.read_csv('experiments/testing-config1-pso-poisson-changev/overlap.csv')
print(df.mean())

df = pd.read_csv('experiments/testing-config1-pso-poisson-changev/sendingRate.csv')
print(df.mean())

# df = pd.read_csv('experiments/testing-config1-mae-turnonmodel-none/overlap.csv')
# print(df.mean())