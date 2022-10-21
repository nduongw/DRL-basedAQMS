from functools import total_ordering
from re import X


class Package:
    
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        
    def setX(self, x):
        self.x = x
        
    def setY(self, y):
        self.y = y
           
    def getInfo(self):
        return self.x, self.y
        