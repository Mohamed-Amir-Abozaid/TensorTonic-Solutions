import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    samples,features = X.shape
    w = np.zeros(features)
    b = 0
    for _ in range(steps):
        z = X @ w + b 
        a = _sigmoid(z)
        error = a - y
        dw = (1/samples) * (X.T @ error)
        db = (1/samples) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db

    return (w,b)
    pass