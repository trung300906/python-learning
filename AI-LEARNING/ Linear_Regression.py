import numpy as np
import matplotlib.pyplot as plt

# Tải dữ liệu
A = np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate.txt", delimiter=',')
print(A[0:4, :])

# Tạo và lưu mảng mẫu
B = np.array([[1, 2, 3.5], [2, 1, 4.6]])
np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/SAVEDATAT.TXT", B, fmt="%.2f", delimiter=' ')
np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/SAVETXT2.txt", A[3:25, :], fmt="%.2f", delimiter=';')

# Hàm tính toán chi phí (cost function)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))

    for i in range(iterations):
        predictions = X.dot(theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot(predictions - y))
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i, :] = theta.T
        
    return theta, cost_history, theta_history

# Hàm hồi quy tuyến tính
def linear_function():
    X = np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate.txt", delimiter=',')
    y = np.copy(A[:, 1])
    X[:, 1] = A[:, 0]
    X[:, 0] = 1
    # Đọc trọng số ban đầu từ file
    initial_theta = np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate_theta.txt", delimiter=',')
    
    # Thực hiện Gradient Descent
    learning_rate = 0.0000000001
    iterations = 1500
    theta, cost_history, theta_history = gradient_descent(X, y, initial_theta, learning_rate, iterations)
    
    # Dự đoán
    predict = X @ theta
    predict = predict * 10000
    np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/answer.txt", ((X @ theta) * 10000), delimiter=',')
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
