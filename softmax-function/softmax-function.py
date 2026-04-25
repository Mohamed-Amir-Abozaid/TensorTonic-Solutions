import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    x_shift = x-np.max(x)
    if x.ndim==1:
        sf = np.exp(x_shift) / np.sum(np.exp(x_shift))
        return sf

    x_shift = x-np.max(x,axis=1,keepdims=True)
    sf = np.exp(x_shift) / np.sum(np.exp(x_shift),axis=1,keepdims=True)

    
    return sf
    pass