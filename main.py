from src.Car import Car
from src.Map import Map
import random
import numpy as np

# random.seed(42)
map = Map()

for i in range(5):
    print(f'Run {i + 1}:')
    map.run()
    print('--------------------------\n')
    