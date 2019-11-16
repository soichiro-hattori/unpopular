import numpy as np
import matplotlib.pyplot as plt

from .target_data import TargetData

class CustomModel(object):
    """A custom model object.

    Args:
        target_data (TargetData): A TargetData instance
    """

    name = "CustomModel"

    def __init__(self, target_data):
        if isinstance(target_data, TargetData):
            self.target_data = target_data
            self.time = target_data.time
        
        self.num_terms = None
        self.m = None
        self.reg = None
        self.reg_matrix = None
        self.params = None
        self.prediction = None

    def set_L2_reg(self, reg):
        """Set the L2-regularization for the custom model.

        Args:
            reg (float): The L2-regularization value.

        """
        self.reg = reg
        self.reg_matrix = reg * np.identity(self.num_terms)

    def predict(self, m=None, params=None, mask=None):
        """Make a prediction for the custom model.

        Args:
            m (Optional[array]): Manually pass the design matrix to use for the prediction.
                Must have dimensions of 
            params (Optional[array]): Manually pass the parameters to use for the prediction.
            mask (Optional[array]): 

        """
        #Unless the user explicitly provides the design matrix or parameters, use the default.
        if m is None:
            m = self.m
        if params is None:
            params = self.params

        if mask is not None:
            m = m[~mask]  # pylint: disable=invalid-unary-operand-type

        prediction = np.dot(m, params)
        self.prediction = prediction
        return prediction
