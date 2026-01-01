import numpy as np

sigmoid = lambda x : 1 / (1 + np.exp(-x))
def loss(X, y, beta, N):
    # X: inputs, y : labels, beta : model parameters, N : # datapoints
    p_hat = sigmoid(np.dot(X, beta))
    return -(1/N)*sum(y*np.log(p_hat) + (1 - y)*np.log(1 - p_hat))

# Here is a NumPy implementation of the derivative of our loss function.

def d_loss(X, y, beta, N):
    # X: inputs, y : labels, beta : model parameters, N : # datapoints
    p_hat = sigmoid(np.dot(X, beta))
    return np.dot(X.T,-(1/N)*(y*(1 - p_hat) - (1 - y)*p_hat))

step_size = .5
n_iter = 500
beta = np.zeros(2) # initial guess for Beta_0 and Beta_1
losses = []
for _ in range(n_iter):
    beta = beta - step_size * d_loss(beta)
    losses.append(loss(beta))