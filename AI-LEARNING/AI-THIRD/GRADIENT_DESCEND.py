import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING")
from funtion import *

raw_data= np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI-THIRD/data.txt", delimiter=',')
np.random.shuffle(raw_data)
y = np.copy(raw_data[:,1])
X = np.copy(raw_data)
X[:,1] = X[:,0]
X[:,0] = 1
X = normalize(X)
theta = np.zeros(np.size(raw_data, 1))
print(theta)
learning_rate = 0.01
iteration = 2000
theta,cost,theta_history = gradient_descent(X, y, theta, learning_rate, iteration)

def linear_regression(x,theta):
    predict = x@theta
    predict = predict * 10000
    
    # Vẽ đồ thị dự đoán
    plt.plot(X[:, 1:], predict / 10000, '-b', label="Dự đoán")
    plt.plot(X[:, 1:], y, 'rx', label="Thực tế")
    plt.title("Dự đoán")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

    # Vẽ đồ thị chi phí theo số lần lặp
    plt.plot(range(iteration), cost, '-r')
    plt.title("Chi phí theo số lần lặp")
    plt.xlabel("Số lần lặp")
    plt.ylabel("Chi phí")
    plt.show()
    
    # Vẽ đồ thị sự thay đổi của Theta
    plt.plot(range(iteration), theta_history[:, 0], '-b', label="Theta 0")
    plt.plot(range(iteration), theta_history[:, 1], '-g', label="Theta 1")
    plt.title("Sự thay đổi của Theta theo số lần lặp")
    plt.xlabel("Số lần lặp")
    plt.ylabel("Giá trị Theta")
    plt.legend()
    plt.show()
linear_regression(X, theta)