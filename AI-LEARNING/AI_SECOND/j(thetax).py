import numpy as np
import sys
sys.path.append("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING")
from funtion import *
raw_data = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_SECOND/data.txt", delimiter=',')
y = np.copy(raw_data[:, 2])
x = np.zeros((np.size(y), np.size(raw_data,1)))
x[:,0] = 1
x[:, 1:] =  np.copy(raw_data[:, 0:2])
Theta = np.array([1,2,5])
print(computeCost(x,y,Theta),compute_cost_vector(x,y,Theta))