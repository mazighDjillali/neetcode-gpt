import numpy as np
from numpy.typing import NDArray
import math 

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        max = np.max(z)
        # Hint: subtract max(z) for numerical stability before computing exp
        normalized_z = z - max
        expo = np.exp(normalized_z)
        return np.round(expo / np.sum(expo),4 )
        
