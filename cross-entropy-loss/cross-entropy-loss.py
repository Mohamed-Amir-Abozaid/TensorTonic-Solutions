import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    samples = y_true.shape[0]
    loss = 0
    for i in range(samples):
        t = y_true[i]
        p = y_pred[i][t]
        loss -= np.log(p)

    return loss/samples
    pass