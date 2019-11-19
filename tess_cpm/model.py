import numpy as np
import lightkurve as lk
from sklearn.model_selection import KFold

from .target_data import TargetData
from .cpm_model import CPM
from .poly_model import PolyModel
from .custom_model import CustomModel


class PixelModel(object):
    """A pixel model object that can store the different model components for the model used in each target pixel
    """

    def __init__(self, target_data, row, col):
        if isinstance(target_data, TargetData):
            self.target_data = target_data
        else:
            return
        self.row = row
        self.col = col
        self.time = self.target_data.time
        self.y = self.target_data.normalized_fluxes[:, row, col]
        self.cpm = None
        self.poly_model = None
        self.custom_model = None
        self.regs = []
        self.reg_matrix = None
        self.m = None
        self.params = None
        self.prediction = None
        self.cpm_prediction = None
        self.poly_model_prediction = None
        self.cpm_subtracted_lc = None
        self.split_time = []
        self.split_fluxes = []
        self.split_prediction = []
        self.split_cpm_prediction = []
        self.split_poly_model_prediction = []
        self.split_custom_model_prediction = []
        self.split_cpm_subtracted_lc = []

    @property
    def models(self):
        return list(filter(bool, [self.cpm, self.poly_model, self.custom_model]))

    def add_cpm_model(
        self,
        exclusion_size=5,
        exclusion_method="closest",
        n=256,
        predictor_method="cosine_similarity",
        seed=None,
    ):
        cpm = CPM(self.target_data)
        cpm.set_target_exclusion_predictors(
            self.row,
            self.col,
            exclusion_size=exclusion_size,
            exclusion_method=exclusion_method,
            n=n,
            predictor_method=predictor_method,
            seed=seed,
        )
        self.cpm = cpm

    def remove_cpm_model(self):
        self.cpm = None

    def add_poly_model(self, scale=2, num_terms=4):
        poly_model = PolyModel(self.target_data)
        poly_model.set_poly_model(scale=scale, num_terms=num_terms)
        self.poly_model = poly_model

    def remove_poly_model(self):
        self.poly_model = None

    def add_custom_model(self, flux):
        custom_model = CustomModel(self.target_data, flux)
        self.custom_model = custom_model
    
    def remove_custom_model(self, flux):
        self.custom_model = None

    def set_regs(self, regs=[], verbose=True):
        if len(regs) != len(self.models):
            print(
                "You need to specify the same number of regularization parameters as the number of models."
            )
            return
        self.regs = regs
        for reg, model in zip(self.regs, self.models):
            if verbose:
                print(f"Setting {model.name}'s regularization to {reg}")
            model.set_L2_reg(reg)
        self._create_reg_matrix()
        self._create_design_matrix()

    def _create_reg_matrix(self):
        r = []
        for mod in self.models:
            r.append(np.repeat(mod.reg, mod.reg_matrix.shape[0]))
        r = np.hstack(r)
        self.reg_matrix = r * np.identity(r.size)

    def _create_design_matrix(self):
        design_matrices = []
        for mod in self.models:
            design_matrices.append(mod.m)
        self.m = np.hstack((design_matrices))

    def fit(self, y=None, m=None, mask=None, save=True):
        if self.regs == []:
            print("Please set the L-2 regularizations first.")
            return
        # self._create_reg_matrix()
        # self._create_design_matrix()
        if y is None:
            print("Fitting using full light curve.")
            y = self.y
        if m is None:
            m = self.m
        if mask is None:
            mask = np.full(y.shape, True)
        y = y[mask]
        m = m[mask]

        a = np.dot(m.T, m) + self.reg_matrix
        b = np.dot(m.T, y)
        params = np.linalg.solve(a, b)
        if save:
            self.params = params
            # Hardcoded! Not ideal.
            if self.cpm is not None:
                self.cpm.params = self.params[: self.cpm.num_predictor_pixels]
            if self.poly_model is not None:
                self.poly_model.params = self.params[self.cpm.num_predictor_pixels :]
        return params

    def predict(self, m=None, params=None, mask=None, save=True):
        if m is None:
            m = self.m
        if params is None:
            params = self.params
        if mask is None:
            mask = np.full(m.shape[0], True)
        m = m[mask]
        prediction = np.dot(m, params)
        # Why does doing self.m give different results...
        # p = np.dot(self.m, self.params)
        # print(np.allclose(p, prediction))
        # print(np.alltrue(p == prediction))
        if save:
            self.prediction = prediction
        return prediction

    def holdout_fit(self, k=10, mask=None):
        if self.regs == []:
            print("Please set the L-2 regularizations first.")
            return

        if mask is None:
            mask = np.full(self.time.shape, True)
        y = self.y[mask]
        m = self.m[mask]

        times = []
        y_tests = []
        m_test_matrix = []
        param_matrix = np.zeros((k, m.shape[1]))

        kf = KFold(k)
        i = 0
        for train, test in kf.split(y):
            y_train, y_test = y[train], y[test]
            m_train, m_test = m[train], m[test]
            times.append(self.time[test])
            y_tests.append(y_test)
            m_test_matrix.append(m_test)
            params = self.fit(y_train, m_train, save=False)
            param_matrix[i] = params
            i += 1
        self.split_fluxes = y_tests
        return (times, y_tests, m_test_matrix, param_matrix)

    def holdout_fit_predict(self, k=10, mask=None, save=True):
        times, y_tests, m_tests, param_matrix = self.holdout_fit(k, mask)
        predictions = [np.dot(m, param) for m, param in zip(m_tests, param_matrix)]
        self.split_prediction = predictions
        self.prediction = np.concatenate(predictions)
        for m, param in zip(m_tests, param_matrix):
            if self.cpm is not None:
                m_cpm, param_cpm = m[:, : self.cpm.num_predictor_pixels], param[: self.cpm.num_predictor_pixels]
                self.split_cpm_prediction.append(np.dot(m_cpm, param_cpm))
            if self.poly_model is not None:
                m_poly, param_poly = m[:, self.cpm.num_predictor_pixels :], param[self.cpm.num_predictor_pixels :]
                self.split_poly_model_prediction.append(np.dot(m_poly, param_poly))
        self.split_cpm_subtracted_lc = [y-cpm for y, cpm, in zip(self.split_fluxes, self.split_cpm_prediction)]
        self.cpm_subtracted_lc = np.concatenate(self.split_cpm_subtracted_lc)
        self.cpm_prediction = np.concatenate(self.split_cpm_prediction)
        if self.poly_model is not None:
            self.poly_model_prediction = np.concatenate(self.split_poly_model_prediction)
        return (times, y_tests, predictions)

    # def _get_hyperparameters(
    #     self, y, rescale=True, k=10, grid_size=30, transit_duration=13
    # ):
    #     """Obtain the regularization hyperparameters by performing cross validation.

    #     Args:
    #         y (array): The target pixel fluxes
    #     """
    #     print(
    #         f"Performing {k}-fold cross validation to obtain regularization parameters."
    #     )

    #     kf = KFold(k)
    #     counter = 0
    #     cdpps = np.zeros((k, grid_size))
    #     for train, test in kf.split(y):
    #         print(f"{counter+1} / {k}")
    #         y_train, y_test = y[train], y[test]
    #         m_train, m_test = self.m[train], self.m[test]

    #         if rescale:
    #             return

    # def _get_params(self, y):
    #     return
