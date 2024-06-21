import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import KFold
from scipy.linalg import block_diag

from .cutout_data import CutoutData
from .cpm_model import CPM
from .poly_model import PolyModel
from .custom_model import CustomModel


class PixelModel(object):
    """A pixel model object that can store the different model components for the model used in each target pixel
    """

    def __init__(self, cutout_data, row, col):
        if isinstance(cutout_data, CutoutData):
            self.cutout_data = cutout_data
        else:
            return
        self.row = row
        self.col = col
        self.time = self.cutout_data.time
        self.raw_flux = self.cutout_data.fluxes[:, row, col]
        self.norm_flux = self.cutout_data.normalized_fluxes[:, row, col]
        self.median = self.cutout_data.flux_medians[row, col]
        self.flux_errors = self.cutout_data.flux_errors[:, row, col]
        self.normalized_flux_errors = self.cutout_data.normalized_flux_errors[:, row, col]
        self.cpm = None
        self.poly_model = None
        self.custom_model = None
        self.regs = []
        self.reg_matrix = None
        self.design_matrix = None
        self.params = None
        self.param_matrix = None
        self.prediction = None
        self.cpm_prediction = None
        self.poly_model_prediction = None
        self.intercept_prediction = None
        self.cpm_subtracted_flux = None
        self.rescaled_cpm_subtracted_flux = None
        self.split_time = []
        self.split_fluxes = []
        self.split_prediction = []
        self.split_cpm_prediction = []
        self.split_poly_model_prediction = []
        self.split_intercept_prediction = []
        self.split_custom_model_prediction = []
        self.split_cpm_subtracted_flux = []
        self.split_rescaled_cpm_subtracted_flux = []

    @property
    def model_components(self):
        return list(filter(bool, [self.cpm, self.poly_model, self.custom_model]))
    
    @property
    def values_dict(self):
        return {
            "raw" : self.raw_flux,
            "normalized_flux" : self.norm_flux,
            "cpm_prediction" : self.cpm_prediction,
            "poly_model_prediction" : self.poly_model_prediction,
            "intercept_prediction" : self.intercept_prediction,
            "cpm_subtracted_flux" : self.cpm_subtracted_flux,
            "rescaled_cpm_subtracted_flux" : self.rescaled_cpm_subtracted_flux
        }
    
    @property
    def split_values_dict(self):
        return {
            "raw" : np.array([(split_flux + 1) * self.median for split_flux in self.split_fluxes], dtype=object),
            "normalized_flux" : np.array(self.split_fluxes, dtype=object),
            "cpm_prediction" : np.array(self.split_cpm_prediction, dtype=object),
            "poly_model_prediction" : np.array(self.split_poly_model_prediction, dtype=object),
            "intercept_prediction" : np.array(self.split_intercept_prediction, dtype=object),
            "cpm_subtracted_flux" : np.array(self.split_cpm_subtracted_flux, dtype=object),
            "rescaled_cpm_subtracted_flux" : np.array(self.split_rescaled_cpm_subtracted_flux, dtype=object)
        }

    def add_cpm_model(
        self,
        exclusion_size=5,
        exclusion_method="closest",
        n=256,
        predictor_method="similar_brightness",
        seed=None,
    ):
        cpm = CPM(self.cutout_data)
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
        poly_model = PolyModel(self.cutout_data)
        poly_model.set_poly_model(scale=scale, num_terms=num_terms)
        self.poly_model = poly_model

    def remove_poly_model(self):
        self.poly_model = None

    def add_custom_model(self, flux):
        custom_model = CustomModel(self.cutout_data, flux)
        self.custom_model = custom_model
    
    def remove_custom_model(self, flux):
        self.custom_model = None

    def set_regs(self, regs=[], verbose=True):
        if len(regs) != len(self.model_components):
            print(
                "You need to specify the same number of regularization parameters as the number of model components."
            )
            return
        self.regs = regs
        for reg, model in zip(self.regs, self.model_components):
            if verbose:
                print(f"Setting {model.name}'s regularization to {reg}")
            model.set_L2_reg(reg)
        self._create_reg_matrix()
        self._create_design_matrix()

    def _create_reg_matrix(self):
        self.reg_matrix = block_diag(*[mod.reg_matrix for mod in self.model_components])

    def _create_design_matrix(self):
        self.design_matrix = np.hstack([mod.m for mod in self.model_components])

    def fit(self, y=None, m=None, mask=None, save=True, verbose=True):
        if self.regs == []:
            print("Please set the L-2 regularizations first.")
            return
        # self._create_reg_matrix()
        # self._create_design_matrix()
        if y is None:
            print("Fitting using full light curve.")
            y = self.norm_flux
        if m is None:
            print("Fitting using full light curve.")
            m = self.design_matrix
        if mask is None:
            mask = np.full(y.shape, True)
        elif mask is not None and verbose:
            print(f"Using user-provided mask. Clipping {np.sum(~mask)} points.")  # pylint: disable=invalid-unary-operand-type
        y = y[mask]
        m = m[mask]

        a = np.dot(m.T, m) + self.reg_matrix
        b = np.dot(m.T, y)
        if verbose:
            print(f"Numpy Defined Condition Number: {np.linalg.cond(a)}")
            # eigvals, eigvecs = np.linalg.eigh(a)
            # eigvals = eigvals[np.nonzero(eigvals)]
            # max_eigval, min_eigval = eigvals.max(), eigvals.min()
            # eigval_ratio = max_eigval / min_eigval
            # print(f"Eigenvalue Ratio Condition Number: {eigval_ratio:.2f} (Max: {max_eigval:.2f}, Min: {min_eigval:.2f})")
        params = np.linalg.solve(a, b)
        if save:
            self.params = params
            # Hardcoded! Not ideal.
            if self.cpm is not None:
                self.cpm.params = self.params[: self.cpm.num_predictor_pixels]
            if self.poly_model is not None:
                self.poly_model.params = self.params[self.cpm.num_predictor_pixels :]
        return params

    def holdout_fit(self, k=10, mask=None, verbose=True):
        if self.regs == []:
            print("Please set the L-2 regularizations first.")
            return

        if mask is None:
            mask = np.full(self.time.shape, True)
        # time = self.time[mask]
        # y = self.norm_flux[mask]
        # m = self.design_matrix[mask]
        time = self.time
        y = self.norm_flux
        m = self.design_matrix

        times = []
        y_tests = []
        m_test_matrix = []
        param_matrix = np.zeros((k, m.shape[1]))

        kf = KFold(k)
        i = 0
        for train, test in kf.split(y):
            y_train, y_test = y[train], y[test]
            m_train, m_test = m[train], m[test]
            mask_train = mask[train]
            times.append(time[test])
            y_tests.append(y_test)
            m_test_matrix.append(m_test)
            params = self.fit(y_train, m_train, mask=mask_train, save=False, verbose=verbose)
            param_matrix[i] = params
            i += 1
        self.split_time = times
        self.split_fluxes = y_tests
        return (times, y_tests, m_test_matrix, param_matrix)

    def holdout_fit_predict(self, k=10, mask=None, save=True, verbose=False):
        self._reset_values()
        times, y_tests, m_tests, param_matrix = self.holdout_fit(k, mask, verbose=verbose)
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
                self.split_intercept_prediction.append(np.multiply(m_poly[:,-1], param_poly[-1]))
        self.split_cpm_subtracted_flux = [y-cpm for y, cpm in zip(self.split_fluxes, self.split_cpm_prediction)]
        # self.split_cpm_subtracted_flux = [y-cpm-param_poly[0] for y, cpm in zip(self.split_fluxes, self.split_cpm_prediction)]  # just to fix plot for presentation

        self.cpm_subtracted_flux = np.concatenate(self.split_cpm_subtracted_flux)
        self.cpm_prediction = np.concatenate(self.split_cpm_prediction)
        if self.poly_model is not None:
            self.poly_model_prediction = np.concatenate(self.split_poly_model_prediction)
            self.intercept_prediction = np.concatenate(self.split_intercept_prediction)
        self.param_matrix = param_matrix
        return (times, y_tests, predictions)

    def _reset_values(self):
        self.split_time = []
        self.split_fluxes = []
        self.split_prediction = []
        self.split_cpm_prediction = []
        self.split_poly_model_prediction = []
        self.split_intercept_prediction = []
        self.split_custom_model_prediction = []
        self.split_cpm_subtracted_flux = []
        self.split_rescaled_cpm_subtracted_flux = []

    def rescale(self):
        # self.split_rescaled_cpm_subtracted_flux = [(flux + 1) * self.median for flux in self.split_cpm_subtracted_flux]
        if self.poly_model is not None:
            self.split_rescaled_cpm_subtracted_flux = [(dt_flux-inter+1) * self.median for dt_flux, inter in zip(self.split_cpm_subtracted_flux, self.split_intercept_prediction)]
        else:
            self.split_rescaled_cpm_subtracted_flux = [(dt_flux+1) * self.median for dt_flux in self.split_cpm_subtracted_flux]
        self.rescaled_cpm_subtracted_flux = np.concatenate(self.split_rescaled_cpm_subtracted_flux)

    def plot_model(self, size_predictors=2):
        return self.cpm.plot_model(size_predictors=size_predictors)

    def summary_plot(self, figsize=(16, 5.5), zeroing=True, show_location=False, size_predictors=10):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 5, hspace=0)

        ax1 = fig.add_subplot(gs[:2, :2])
        ax2 = fig.add_subplot(gs[0, 2:5])
        ax3 = fig.add_subplot(gs[1, 2:5])

        self.cpm._plot_model_onto_axes(ax1, size_predictors=size_predictors)
        ax1.set_xticks(np.arange(0, 100, 10))
        ax1.set_yticks(np.arange(0, 100, 10))
        ax1.set_xlabel("Pixel Column Number", fontsize=15)
        ax1.set_ylabel("Pixel Row Number", fontsize=15)
        ax1.tick_params(labelsize=15)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.5)
        
        ax2.plot(self.time, self.norm_flux, "k.", ms=5, alpha=0.2, 
                 label="Normalized Flux")
        if show_location:
                ax2.text(x=0.98, y=0.95, s=f"[{self.row},{self.col}]", 
                        ha='right', va='top', transform=ax2.transAxes)
        
        if zeroing:
            y_ax2 = self.cpm_prediction + self.intercept_prediction
            y_ax3 = self.cpm_subtracted_flux - self.intercept_prediction
        else:
            y_ax2 = self.cpm_prediction
            y_ax3 = self.cpm_subtracted_flux

        ax2.plot(self.time, y_ax2, "C3-", lw=1.5, alpha=0.7, label="CPM Prediction")
        ax3.plot(self.time, y_ax3, "k.", ms=5, alpha=0.2)

        ax2.legend(markerscale=2, fontsize=10, edgecolor="k")
        ax2.tick_params(axis="y", labelsize=8)
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(1.5)
        ax3.tick_params(axis="y", labelsize=15)
        ax3.tick_params(axis="y", labelsize=8)

        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(1.5)

        ax3.set_xlabel("Time [BJD - 2457000]", fontsize=15)
        fig.text(0.405, 2/3, "Normalized Flux", fontsize=12, rotation="vertical", va="center")
        fig.text(0.405, 1/3, "De-trended Flux", fontsize=12, rotation="vertical", va="center")


        plt.show()
        return fig, [ax1, ax2, ax3]