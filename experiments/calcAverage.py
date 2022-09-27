import pandas as pd

df = pd.read_csv('experiments/testing-config7-model-poisson/sendingRate.csv')
print(df.mean())