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
    samples, features = X.shape
    w = np.zeros(features) #(f,)
    b = 0
    # (s,f)*(f,) -> (s,)
    for _ in range(steps):
        z = X @ w + b
        y_p = _sigmoid(z) #-> (s,)
        #binary cross entropy 
        # y -> (samples,)
        dw = (1/samples)* X.T @ (y_p - y)
        db = (1/samples)* np.sum(y_p - y)

        w = w - lr*dw
        b = b - lr*db

    return w,b
    pass