import numpy as np
import matplotlib.pyplot as plt

from .cutout_data import CutoutData

class CustomModel(object):
    """A custom model object.

    Args:
        cutout_data (cutoutData): A cutoutData instance
    """

    name = "CustomModel"

    def __init__(self, cutout_data, flux=None):
        if isinstance(cutout_data, CutoutData):
            self.cutout_data = cutout_data
            self.time = cutout_data.time
        
        self.num_terms = None
        self.m = None
        self.reg = None
        self.reg_matrix = None
        self.params = None
        self.prediction = None

        if flux is not None:
            self.set_des_mat(flux)

    def set_des_mat(self, flux):
            if flux.size != self.time.size:
                print("The custom model lightcurve must be the same length as the dataset.")
                return
            else:
                self.m = flux.reshape((-1, 1))
                self.num_terms = self.m.shape[1]

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
