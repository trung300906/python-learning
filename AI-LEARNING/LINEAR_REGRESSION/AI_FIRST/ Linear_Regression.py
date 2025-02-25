import numpy as np
import matplotlib.pyplot as plt
import time
import sys 
sys.path.append("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING")
from funtion import *

A = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/univariate.txt", delimiter=',')
# Hàm hồi quy tuyến tính
def linear_function():
    X = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/univariate.txt", delimiter=',')
    y = np.copy(A[:, 1])
    X[:, 1] = A[:, 0]
    X[:, 0] = 1
    # Đọc trọng số ban đầu từ file
    initial_theta = np.zeros(np.size(A, 1))
    
    # Thực hiện Gradient Descent
    learning_rate = 0.01
    iterations = 5000
    theta, cost_history, theta_history = gradient_descent(X, y, initial_theta, learning_rate, iterations)
    
    # Dự đoán
    predict = X @ theta
    predict = predict * 10000
    np.savetxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/answer.txt", ((X @ theta) * 10000), delimiter=',')
    print('%d người : %.2f$' % (X[0, 1] * 10000, predict[0]))
    
    # Vẽ đồ thị dự đoán
    plt.plot(X[:, 1:], predict / 10000, '-b', label="Dự đoán")
    plt.plot(X[:, 1:], y, 'rx', label="Thực tế")
    plt.title("Dự đoán")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
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

linear_function()
