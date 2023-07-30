from typing import List

import numpy as np


def compute_topk(predictions: np.array) -> List[float]:
    """
    Computes the topk accuracy of a boolean np array

    Args:
        predictions: boolean np.array of shape batch_size x k with correctness of each
            prediction

    Returns:
        List of floats denoting the top-k accuracies
    """

    topk = [np.mean(predictions[:, 0])]
    for k in range(1, predictions.shape[1]):
        topk.append(topk[-1] + np.mean(predictions[:, k]))
    return topk
