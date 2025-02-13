import numpy as np
import time
def loadtxt(path_to_file, sign):
    try:
        raw_data= np.loadtxt(path_to_file, delimiter=sign)
        x = np.zeros((np.size(raw_data, 0), np.size(raw_data, 1)))
        x[:, 0] = 1
        x[:,1:] = raw_data[:,:-1]
        y = raw_data[:,-1]
        return x, y, raw_data
    except:
        return 0
def savetxt(path_to_file, matrix, sign):
    np.savetxt(path_to_file, matrix, delimiter=sign)

def predict_function(X, Theta):
    return X@Theta

def computeCost(X,y,Theta):
    predicted = X@Theta
    sqr_error = (predicted - y)**2
    sum_error = np.sum(sqr_error)
    m = np.size(y)
    J = (1/(2*m))*sum_error
    return J

def compute_cost_vector(X,y,Theta):
    error = predict_function(X, Theta) - y
    m = np.size(y)
    J = (1/(2*m)) * (np.transpose(error)@error)
    return J
# Gradient Descent

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, np.size(X, 1)))
    for i in range(iterations):
        printProgressBar(i,iterations)
        time.sleep(0.0003)
        predictions = X@theta
        theta = theta - (learning_rate / m) * ((np.transpose(X)) @ (predictions - y))
        cost_history[i] = compute_cost_vector(X, y, theta)
        theta_history[i, :] = np.transpose(theta)
        
    return theta, cost_history, theta_history

def gradient_descent_handmade(X,y,theta, learning_rate, iterations):
    m = len(y)
    x_t = np.transpose(X)
    theta_history = np.zeros((iterations,2))
    cost_history = np.zeros(iterations)
    #error = compute_cost_vector(X,y,theta)
    for i in range(iterations):
        printProgressBar(i,iterations)
        time.sleep(0.0003)
        predictions = X@theta
        theta = theta - (learning_rate / m)*(x_t @ (predictions - y))
        theta_history[i,:] = np.transpose(theta)
        cost_history[i] = compute_cost_vector(X,y,theta)
    
    return theta, cost_history, theta_history
        
def normal_equation(X, y):
    return np.linalg.pinv(np.transpose(X)@X)@(np.transpose(X)@y)

def printProgressBar (iteration, total, suffix = ''):
    percent = ("{0:." + str(1) + "f}").format(100 * ((iteration+1) / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '=' * filledLength + '-' * (50- filledLength)
    print('\rTraining: |%s| %s%%' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if(percent==iteration):
        print()

def normalize(x):
    n = np.copy(x)
    n[0,0] = 100
    s = np.std(n, 0, dtype = np.float64)
    mu = np.mean(x, 0)
    n=(n-mu)/s
    n[:,0]=1
    return n

def adam_optimizer(f_grad, x_init, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    """
    Adam optimizer từ đầu.
    f_grad: Hàm trả về gradient của f(x)
    x_init: Điểm khởi tạo
    learning_rate: Tốc độ học
    beta1, beta2: Hệ số trung bình động
    epsilon: Hằng số tránh chia cho 0
    max_iter: Số vòng lặp tối đa
    """
    x = x_init
    m, v = 0, 0  # Khởi tạo moment bậc 1 và 2
    t = 0        # Đếm số bước
    
    for t in range(1, max_iter + 1):
        grad = f_grad(x)  # Tính gradient
        
        m = beta1 * m + (1 - beta1) * grad  # Cập nhật moment bậc 1
        v = beta2 * v + (1 - beta2) * (grad ** 2)  # Cập nhật moment bậc 2
        
        m_hat = m / (1 - beta1 ** t)  # Hiệu chỉnh bias moment bậc 1
        v_hat = v / (1 - beta2 ** t)  # Hiệu chỉnh bias moment bậc 2
        
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Cập nhật tham số
        
    return x
