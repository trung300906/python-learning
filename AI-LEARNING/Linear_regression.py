import numpy as np 
import matplotlib.pyplot as plt
A = np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate.txt", delimiter=',')
print(A[0:4, :])

B = np.array([[1,2,3.5],[2,1,4.6]])
np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/SAVEDATAT.TXT",B, fmt="%.2f", delimiter=' ')
np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/SAVETXT2.txt", A[3:25,:], fmt="%.2f", delimiter=';')

def linear_function():
    X=np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate.txt", delimiter=',')
    y = np.copy(A[:,1])
    X[:,1] = A[:, 0]
    X[:,0] = 1
    Theta=np.loadtxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/univariate_theta.txt", delimiter=',')
    predict = X@Theta
    predict = predict * 10000
    np.savetxt("/run/media/trung/linuxandwindows/code/python-learning/AI-LEARNING/answer.txt", ((X@Theta)*10000)  , delimiter =',')
    print('%d người : %.2f$' %(X[0,1]*10000,predict[0]))
    plt.plot(X[:,1:],predict/10000,'-b')
    plt.plot(X[:,1:],y,'rx')
    plt.title("predict")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show() 
linear_function()