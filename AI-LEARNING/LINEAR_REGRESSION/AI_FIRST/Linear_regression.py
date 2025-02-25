import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
A = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/univariate.txt", delimiter=',')
#print(A[0:4, :])

B = np.array([[1,2,3.5],[2,1,4.6]])
np.savetxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/SAVEDATAT.TXT",B, fmt="%.2f", delimiter=' ')
np.savetxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/SAVETXT2.txt", A[3:25,:], fmt="%.2f", delimiter=';')

def linear_regression_univariate():
    X=np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/univariate.txt", delimiter=',')
    y = np.copy(A[:,1])
    X[:,1] = A[:, 0]
    X[:,0] = 1
    Theta=np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/univariate_theta.txt", delimiter=',')
    predict = X@Theta
    predict = predict * 10000
    np.savetxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/answer.txt", ((X@Theta)*10000)  , delimiter =',')
    print('%d người : %.2f$' %(X[0,1]*10000,predict[0]))
    plt.plot(X[:,1:],predict/10000,'-b', label="DỰ ĐOÁN")
    plt.plot(X[:,1:],y,'rx', label="thực tế")
    plt.title("predict")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()
linear_regression_univariate()

def linear_regression_mulunivariate():
    raw_data = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/multivariate.txt", delimiter=',')
    initial_theta = np.loadtxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/multivariate_theta.txt", delimiter=',')
    y = np.copy(raw_data[:,-1])
    x = np.zeros((np.size(raw_data, 0), np.size(raw_data, 1)))
    x[:,0] = 1
    x[:,1:2] = raw_data[:,0:1]
    predict = x@initial_theta
    np.savetxt("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/AI_FIRST/multivariate_predict.txt", predict, delimiter=',')

    
    # Tạo đồ thị 3D bằng scatter plot
    X = raw_data[:, 1]  # Đặc trưng thứ nhất
    Y = raw_data[:, 2]  # Đặc trưng thứ hai
    Z = predict  # Giá trị dự đoán

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ điểm dữ liệu gốc (màu xanh)
    ax.scatter(X, Y, y, color='blue', label='Actual Data')  
    
    # Vẽ điểm dữ liệu dự đoán (màu đỏ)
    ax.scatter(X, Y, Z, color='red', label='Predicted Data')  

    # Thiết lập tiêu đề và nhãn
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target Value")
    ax.set_title("Multivariate Linear Regression Scatter Plot")
    ax.legend()

    plt.legend()
    plt.show()

linear_regression_mulunivariate()

