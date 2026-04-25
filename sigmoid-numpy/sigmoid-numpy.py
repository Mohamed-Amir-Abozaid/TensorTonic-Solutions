import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    seg = 1 / (1 + np.exp(-x))
    return seg
    pass