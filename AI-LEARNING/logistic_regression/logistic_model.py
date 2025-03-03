import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/logistic_regression')
from function import *
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go


raw_data = load_txt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/logistic_regression/data.txt")
theta = load_txt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/logistic_regression/theta.txt")
print("theta: ",theta)
x = np.zeros((np.size(raw_data, 0), np.size(raw_data, 1)))
x[:, 0] = 1
x[:, 1:] = raw_data[:, :-1] 
y = raw_data[:, 1].astype(int)
print("x: ", x[:10])
print("y: ",y[:10])

# write_to_graph(x, y,"Biểu đồ dữ liệu Logistic Regression", "Giá trị X", "Nhãn Y")
def logistic_regression(x, y, theta):
    iterations = 200000
    learning_rate = 0.1
    theta , cost_history= gradient_descent(x, y, theta, learning_rate, iterations)
    print("\ntheta: ",theta)

    predicted_prob = predict(x, theta)  # Dự đoán xác suất
    predicted = predict(x, theta)
    print("predicted: ", predicted)
    
    # Đánh giá mô hình
    accuracy_score = accuracy(y, predicted)
    precision_score = precision(y, predicted)
    recall_score = recall(y, predicted)
    f1 = f1_score(y, predicted)
    print(f"Accuracy: {accuracy_score:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1-score: {f1:.4f}")
    # ROC Curve & AUC
    tpr, fpr = roc_curve(y, predicted_prob)
    auc_score = auc(fpr, tpr)

    print(f"AUC: {auc_score:.4f}")

    # Vẽ đường ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    write_to_graph(x, y, predicted, "Biểu đồ dữ liệu Logistic Regression", "Giá trị X", "Nhãn Y")
logistic_regression(x, y, theta)