import numpy as np
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