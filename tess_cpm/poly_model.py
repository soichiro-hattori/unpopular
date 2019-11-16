import numpy as np
import matplotlib.pyplot as plt

from .target_data import TargetData


class PolyModel(object):
    """A polynomial model object.

    Args:
        target_data (TargetData): A TargetData instance
    """

    name = "PolyModel"

    def __init__(self, target_data):
        if isinstance(target_data, TargetData):
            self.target_data = target_data
            self.time = target_data.time
            self.normalized_time = (
                self.time - (self.time.max() + self.time.min()) / 2
            ) / (self.time.max() - self.time.min())

        self.scale = None
        self.input_vector = None
        self.num_terms = None
        self.m = None
        self.reg = None
        self.reg_matrix = None
        self.params = None
        self.prediction = None

    def set_poly_model(self, scale=2, num_terms=4):
        """Set the polynomial model parameters. 
        
        The polynomial model is used to capture long term trends in the data
        believed to be signal and not background noise (such as supernova lightcurves). This method is essentially 
        calling the ``numpy.vander()`` method.

        Args:
            scale (Optional[float]): Sets the min and max value of the input vector to pass to ``numpy.vander``.
                The larger this value, the more flexibility the polynomial model will have for a given number of powers.
            poly_terms (Optional[int]): Specify the number of "powers" to use in the polynomial model.
                As the first power is a constant, the highest power is ``poly_terms - 1``. 

        """
        self.scale = scale
        self.input_vector = scale * self.normalized_time
        self.num_terms = num_terms
        self.m = np.vander(self.input_vector, N=num_terms, increasing=True)

    def set_L2_reg(self, reg):
        """Set the L2-regularization for the polynomial model.

        Args:
            reg (float): The L2-regularization value.

        """
        self.reg = reg
        self.reg_matrix = reg * np.identity(self.num_terms)

    def predict(self, m=None, params=None, mask=None):
        """Make a prediction for the polynomial model.

        Args:
            m (Optional[array]): Manually pass the design matrix to use for the prediction.
                Must have dimensions of 
            params (Optional[array]): Manually pass the parameters to use for the prediction.
            mask (Optional[array]): 

        """
        # Unless the user explicitly provides the design matrix or parameters, use the default.
        if m is None:
            m = self.m
        if params is None:
            params = self.params

        if mask is not None:
            m = m[~mask]  # pylint: disable=invalid-unary-operand-type

        prediction = np.dot(m, params)
        self.prediction = prediction
        return prediction
