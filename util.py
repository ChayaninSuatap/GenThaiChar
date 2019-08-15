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
        for class_i ,subdir in enumerate(os.listdir(dirname)[:2]):
            fulldirpath = dirname + '/' + subdir
            for fn in os.listdir(fulldirpath):
                img = Image.open(fulldirpath + '/' + fn).convert('L')
                img = np.array(img)
                # img = np.expand_dims(img, axis=3)
                xs.append(img)
                ys.append(class_i)
        return np.array(xs), np.array(ys)
    

if __name__ == '__main__':
    o = DatasetLoader()
    o.load_data()