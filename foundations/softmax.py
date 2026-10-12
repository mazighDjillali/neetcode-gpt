import numpy as np
from numpy.typing import NDArray
import math 

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        normalized_z = z - np.max(z)
        expo = np.exp(normalized_z)
        return np.round(expo / np.sum(expo),4 )
        
