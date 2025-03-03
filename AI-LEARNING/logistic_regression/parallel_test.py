import dpnp as np
from numba_dppy import dpjit
import matplotlib.pyplot as plt

# Hàm load dữ liệu sử dụng numpy (CPU) rồi chuyển sang dpnp
def load_txt(file_path):
    import numpy as onp  # Sử dụng numpy thông thường để load file
    data_cpu = onp.loadtxt(file_path, delimiter=' ')
    return np.array(data_cpu)

# Hàm sigmoid chạy trên GPU
@dpjit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính cost chạy trên GPU
@dpjit
def compute_cost(X, y, theta):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5
    cost = (1 / m) * (-np.dot(y.T, np.log(h + epsilon)) - np.dot((1 - y).T, np.log(1 - h + epsilon)))
    return cost

# Hàm tính gradient chạy trên GPU
@dpjit
def compute_gradient(X, y, theta):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (h - y)) / m
    return grad

# Gradient descent với tính toán song song trên GPU
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        grad = compute_gradient(X, y, theta)
        theta = theta - learning_rate * grad
        cost_history[i] = compute_cost(X, y, theta)
        if i % 1000 == 0:
            # Chuyển cost về CPU để in ra
            print(f"Iteration {i}, Cost: {cost_history[i].get()}")
    return theta, cost_history

def logistic_regression(data_path, theta_path):
    # Load dữ liệu và theta từ file
    raw_data = load_txt(data_path)
    theta = load_txt(theta_path)
    
    m, n = raw_data.shape
    # Giả sử dữ liệu có (n-1) features và nhãn nằm ở cột cuối
    X = np.zeros((m, n))
    X[:, 0] = 1            # Thêm bias term
    X[:, 1:] = raw_data[:, :-1]
    y = raw_data[:, -1]
    y = y.astype(np.int32)
    
    iterations = 10000    # Số vòng lặp
    learning_rate = 0.1   # Tốc độ học
    
    # Huấn luyện mô hình logistic regression
    theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
    print("Theta tối ưu:", theta.get())
    
    # Dự đoán xác suất và nhãn
    predicted_prob = sigmoid(np.dot(X, theta))
    predicted = (predicted_prob >= 0.5).astype(np.int32)
    
    # Tính độ chính xác
    accuracy = np.sum(predicted == y) / y.shape[0] * 100
    print("Độ chính xác:", accuracy.get())
    
    # Vẽ đồ thị cost theo iterations (chuyển về CPU để vẽ với matplotlib)
    plt.figure()
    plt.plot(cost_history.get())
    plt.title("Cost History")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    # Vẽ đồ thị dữ liệu thật và dự đoán (giả sử chỉ sử dụng feature chính ở X[:,1])
    plt.figure()
    plt.scatter(X[:, 1].get(), y.get(), color='red', marker='x', label='Actual y')
    plt.scatter(X[:, 1].get(), predicted.get(), color='blue', marker='o', label='Predicted y')
    plt.title("Actual vs Predicted")
    plt.xlabel("Feature X")
    plt.ylabel("Label y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Cập nhật đường dẫn file dữ liệu và file theta cho phù hợp
    data_path = "/path/to/your/data.txt"
    theta_path = "/path/to/your/theta.txt"
    logistic_regression(data_path, theta_path)
