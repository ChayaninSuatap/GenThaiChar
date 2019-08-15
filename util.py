import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DatasetLoader:
    def __init__(self):
        pass
    
    def load_data(self, dirname='mid_level'):
        xs=[]
        ys=[]
        for class_i ,subdir in enumerate(os.listdir(dirname)):
            fulldirpath = dirname + '/' + subdir
            for fn in os.listdir(fulldirpath):
                img = Image.open(fulldirpath + '/' + fn).convert('L')
                img = np.array(img)
                img = 256 - img
                xs.append(img)
                ys.append(class_i)
        return np.array(xs), np.array(ys)
    

if __name__ == '__main__':
    o = DatasetLoader()
    o.load_data()