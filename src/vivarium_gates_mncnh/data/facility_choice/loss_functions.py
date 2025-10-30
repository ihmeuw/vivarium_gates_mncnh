import numpy as np
from scipy.special import xlogy


def elementwise_log_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_dim = len(y_true.shape)
    if n_dim == 1:
        # Each entry represents p for a Bernoulli random variable
        loss = -(xlogy(y_true, y_pred) + xlogy(1 - y_true, 1 - y_pred))
    else:
        # Each row represents p_1,...,p_k for a categorical distribution
        loss = -xlogy(y_true, y_pred)
    return loss


# Modeled after scikit-learn's log_loss function:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
def log_loss(y_true, y_pred, weights=None, normalize=True, epsilon=1e-15):
    y_true, y_pred = map(np.asarray, (y_true, y_pred))
    losses = elementwise_log_loss(y_true, y_pred, epsilon)
    if losses.ndim == 2:
        losses = losses.sum(axis=1)
    if normalize:
        loss = np.average(losses, weights=weights)
    elif weights is not None:
        loss = np.dot(losses, weights)
    else:
        loss = losses.sum()
    return loss
