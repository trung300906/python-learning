import numpy as np
import matplotlib.pyplot as plt

def load_txt(file_path):
	data = np.loadtxt(file_path, delimiter=' ')
	return data

def write_to_graph(X, y, predicted, title, xlabel, ylabel):
    # Chọn cột đầu tiên của X để vẽ (giả sử chỉ có 1 feature chính)
    x_vals = X[:, 1] if X.shape[1] > 1 else X[:, 0]

    # Chia thành 2 nhóm dựa trên y
    x1 = x_vals[y == 1]  # Các điểm có y = 1
    x0 = x_vals[y == 0]  # Các điểm có y = 0

    # Dự đoán nhãn (chuyển từ xác suất sang 0 hoặc 1)
    #predicted_labels = (predicted >= 0.5).astype(int)

    # Vẽ dữ liệu thật
    plt.scatter(x0, np.zeros_like(x0), color='red', marker='x', label='Actual: y = 0')
    plt.scatter(x1, np.ones_like(x1), color='blue', marker='o', label='Actual: y = 1')

    # Vẽ điểm dự đoán (đặt gần 0 hoặc 1)
    plt.scatter(x_vals, predicted, color='green', marker='.', label='Predicted')

    # Thiết lập thông tin đồ thị
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def printProgressBar (iteration, total, suffix = ''):
    percent = ("{0:." + str(1) + "f}").format(100 * ((iteration+1) / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '=' * filledLength + '-' * (50- filledLength)
    print('\rTraining: |%s| %s%%' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if(percent==iteration):
        print()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(X @ theta)

def compute_cost(X, y, theta):
    m = len(y)
    h = predict(X, theta)
    epsilon = 1e-5
    J = (1 / m) * (-y.T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return J

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        printProgressBar(i, iterations) # Hiển thị tiến trình
        h = predict(X, theta)
        theta = theta - (learning_rate / m) * (X.T @ (h - y))
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

def predict_probability(X, theta):
    return sigmoid(X @ theta)

def predict_class(X, theta, threshold=0.5):
    return predict_probability(X, theta) >= threshold


def accuracy(y_true, y_pred):
    """
    Tính độ chính xác của mô hình bằng cách so sánh nhãn thực tế và nhãn dự đoán.
    
    Parameters:
    - y_true: numpy array chứa nhãn thực tế (0 hoặc 1).
    - y_pred: numpy array chứa nhãn dự đoán (0 hoặc 1).
    
    Returns:
    - Độ chính xác (float)
    """
    # Đảm bảo y_pred là nhãn 0 hoặc 1
    y_pred = (y_pred >= 0.5).astype(int)  

    # Kiểm tra kích thước y_true và y_pred có khớp nhau không
    if y_true.shape != y_pred.shape:
        raise ValueError("Kích thước của y_true và y_pred không khớp nhau!")

    return np.sum(y_true == y_pred) / len(y_true) * 100

#specs

def confusion_matrix(y_true, y_pred):
    y_pred = (y_pred >= 0.5).astype(int)  
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def precision(y_true, y_pred):
    TP, _, FP, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r / (p + r)) if (p + r) != 0 else 0

def roc_curve(y_true, y_prob):
    thresholds = np.sort(np.unique(y_prob))  
    tpr = np.zeros_like(thresholds, dtype=float)
    fpr = np.zeros_like(thresholds, dtype=float)
    for i, threshold in enumerate(thresholds):
        y_pred = y_prob >= threshold
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        tpr[i] = TP / (TP + FN)
        fpr[i] = FP / (FP + TN)
    return tpr, fpr

def auc(fpr, tpr):
    if fpr[0] > fpr[-1]:  # Đảo ngược nếu cần
        fpr, tpr = fpr[::-1], tpr[::-1]
    return np.trapezoid(tpr, fpr)

def print_confusion_matrix(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("                Predicted")
    print("              0         1")
    print("Actual 0   {:>8}   {:>8}".format(TN, FP))
    print("       1   {:>8}   {:>8}".format(FN, TP))
