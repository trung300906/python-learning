import numpy as np
import sys
sys.path.append("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/LINEAR_REGRESSION")
import time
from funtion import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go

def linear_regression_multivarite():
    x, y, raw_data=loadtxt('/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/LINEAR_REGRESSION/AI-FOUR/data.txt',',')
    #np.random.shuffle(raw_data)
    # thetaa=normal_equation(x,y)
    Theta = np.zeros(np.size(raw_data,1))
    print(Theta)
    x = normalize(x)
    iterations=2000
    Theta, cost_history, theta_history = gradient_descent(x,y, Theta ,learning_rate=0.1, iterations=2000)
    print(Theta)
    predict = predict_function(x, Theta)
    X = raw_data[:, 0]
    Y = raw_data[:, 1]
    Z = predict
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,raw_data[:, -1], color='blue', label='actual data')
    ax.scatter(X,Y,Z, color='red', label='predict_normalize')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('normalize_data_training')
    plt.legend()
    plt.show()

    # Vẽ đồ thị chi phí theo số lần lặp
    plt.plot(range(iterations), cost_history, '-r')
    plt.title("Chi phí theo số lần lặp")
    plt.xlabel("Số lần lặp")
    plt.ylabel("Chi phí")
    plt.show()
    
    # Vẽ đồ thị sự thay đổi của Theta
    plt.plot(range(iterations), theta_history[:, 0], '-b', label="Theta 0")
    plt.plot(range(iterations), theta_history[:, 1], '-g', label="Theta 1")
    plt.title("Sự thay đổi của Theta theo số lần lặp")
    plt.xlabel("Số lần lặp")
    plt.ylabel("Giá trị Theta")
    plt.legend()
    plt.show()
linear_regression_multivarite() 