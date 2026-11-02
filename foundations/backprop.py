import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        z = np.dot(x,w) + b 
        y_hat = 1 / (1 + np.exp(-z))

        error = y_hat - y_true 
        sigmoid_der = y_hat *(1 - y_hat )
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        grad_w = error * sigmoid_der * x
        grad_b = error * sigmoid_der 

        return (np.round(grad_w, 5), np.round(grad_b,5))