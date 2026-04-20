import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n_samples = len(y_true)
    cr_prob = y_pred[np.arange(n_samples) , y_true]
    loss = -np.mean(np.log(cr_prob))
    return loss
    
    pass