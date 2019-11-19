import numpy as np
import matplotlib.pyplot as plt

from .target_data import TargetData
from .model import PixelModel
from .cpm_model import CPM
from .poly_model import PolyModel


class Source(object):
    """The main interface to interact with both the data and models for a TESS source

    """

    def __init__(self, path, remove_bad=True, verbose=True):
        self.target_data = TargetData(path, remove_bad, verbose)
        self.time = self.target_data.time
        self.aperture = None
        self.models = None
        self.fluxes = None
        self.predictions = None
        self.detrended_lcs = None
        self.split_times = None
        self.split_predictions = None
        self.split_fluxes = None
        self.split_detrended_lcs = None

    def set_aperture(self, rowrange=[49, 52], colrange=[49, 52]):
        self.models = []
        self.fluxes = []
        apt = np.full(self.target_data.fluxes[0].shape, False)
        print("Assuming you're interested in the central set of pixels")
        apt[rowrange[0]:rowrange[1], colrange[0]:colrange[1]] = True

        self.aperture = apt
        for row in range(rowrange[0], rowrange[1]):
            row_models = []
            row_fluxes = []
            for col in range(colrange[0], colrange[1]):
                row_models.append(PixelModel(self.target_data, row, col))
                row_fluxes.append(self.target_data.normalized_fluxes[:, row, col])
            self.models.append(row_models)
            self.fluxes.append(row_fluxes)

    def add_cpm_model(self, exclusion_size=5,
        exclusion_method="closest",
        n=256,
        predictor_method="cosine_similarity",
        seed=None,):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.add_cpm_model(exclusion_size, exclusion_method, n, predictor_method, seed)

    def add_poly_model(self, scale=2, num_terms=4):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.add_poly_model(scale, num_terms)

    def add_custom_model(self, flux):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.add_custom_model(flux)

    def set_regs(self, regs=[], verbose=False):
        if self.models is None:
                print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.set_regs(regs, verbose)

    def holdout_fit_predict(self, k=10, mask=None):
        # if self.models is None:
        #     print("Please set the aperture first.")
        predictions = []
        fluxes = []
        detrended_lcs = []
        for row_models in self.models:
            row_predictions = []
            row_fluxes = []
            row_detrended_lcs = []
            for model in row_models:
                times, flux, pred = model.holdout_fit_predict(k, mask)
                row_fluxes.append(flux)
                row_predictions.append(pred)
                # row_detrended_lcs.append(flux - pred)
            fluxes.append(row_fluxes)
            predictions.append(row_predictions)
            detrended_lcs.append(row_detrended_lcs)
        self.split_times = times
        self.split_fluxes = fluxes
        self.split_predictions = predictions
        self.split_detrended_lcs = detrended_lcs
        return (times, fluxes, predictions)

    def plot_cutout(self, rowrange=None, colrange=None, l=10, h=90, show_aperture=False):
        if rowrange is None:
            rows = slice(0, self.target_data.cutout_sidelength)
        else:
            rows = slice(rowrange[0], rowrange[1]-1)

        if colrange is None:
            cols = slice(0, self.target_data.cutout_sidelength)
        else:
            cols = slice(colrange[0], colrange[1]-1)
        median_image = self.target_data.flux_medians[rows, cols]
        plt.imshow(
            median_image,
            origin="lower",
            vmin=np.percentile(median_image, l),
            vmax=np.percentile(median_image, h),
        )
        if show_aperture:
            plt.imshow(np.ma.masked_where(self.aperture == False, self.aperture), origin='lower', cmap='binary', alpha=0.7)

    def plot_pixel(self, row=None, col=None, loc=None):
        """Plot the data (lightcurve) for a specified pixel.
        """
        flux = self.target_data.fluxes[:, row, col]
        plt.plot(self.target_data.time, flux, ".")

    def plot_pix_by_pix(self, split=True, data_type="raw", figsize=(12, 8), thin=1):
        if self.split_predictions is None:
            self.holdout_fit_predict()
        rows = np.arange(len(self.split_predictions))
        cols = np.arange(len(self.split_predictions[0]))
        fig, axs = plt.subplots(rows.size, cols.size, sharex=True, sharey=True, figsize=figsize)
        for r in rows:
            for c in cols:
                ax = axs[rows[-1] - r, c]  # Needed to flip the rows so that they match origin='lower' setting
                if split:
                    if data_type == "raw":
                        yy = self.models[r][c].split_fluxes
                    elif data_type == "prediction":
                        yy = self.models[r][c].split_prediction
                    elif data_type == "cpm_prediction":
                        yy = self.models[r][c].split_cpm_prediction
                    elif data_type == "poly_model_prediction":
                        yy = self.models[r][c].split_poly_model_prediction
                    # elif data_type == "detrended_lc":
                    #     yy = self.models[r][c].split_detrended_lcs
                    elif data_type == "cpm_subtracted_lc":
                        yy = self.models[r][c].split_cpm_subtracted_lc
                    for time, y in zip(self.split_times, yy):
                        ax.plot(time[::thin], y[::thin], '.')
                else:
                    if data_type == "raw":
                        y = self.models[r][c].y
                    elif data_type == "prediction":
                        y = self.models[r][c].prediction
                    elif data_type == "cpm_prediction":
                        y = self.models[r][c].cpm_prediction
                    elif data_type == "poly_model_prediction":
                        y = self.models[r][c].poly_model_prediction
                    elif data_type == "cpm_subtracted_lc":
                        y = self.models[r][c].cpm_subtracted_lc
                    # elif data_type == "detrended_lc":
                    #     y = np.concatenate(self.models[r][c].split_detrended_lcs)
                    ax.plot(self.time[::thin], y[::thin], '.', c='k')
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # def plot_pix_predictions(self, split=True):
    #     if self.split_predictions is None:
    #         self.holdout_fit_predict()
    #     rows = np.arange(len(self.split_predictions))
    #     cols = np.arange(len(self.split_predictions[0]))
    #     fig, axs = plt.subplots(rows.size, cols.size, sharex=True, sharey=True, figsize=(12, 8))
    #     for r in rows:
    #         for c in cols:
    #             ax = axs[rows[-1] - r, c]
    #             if split:
    #                 for time, chunk in zip(self.split_times, self.split_predictions[r][c]):
    #                     ax.plot(time, chunk, '.')
    #             else:
    #                 ax.plot(self.time, np.concatenate(self.split_predictions[r][c]), '.')
    #     fig.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()

    # def plot_pix_detrended_lcs(self, split=True):
    #     if self.split_predictions is None:
    #         self.holdout_fit_predict()
    #     rows = np.arange(len(self.split_predictions))
    #     cols = np.arange(len(self.split_predictions[0]))
    #     fig, axs = plt.subplots(rows.size, cols.size, sharex=True, sharey=True, figsize=(12, 8))
    #     for r in rows:
    #         for c in cols:
    #             ax = axs[rows[-1] - r, c]  # Needed to flip the rows so that they match origin='lower' setting
    #             if split:
    #                 for time, chunk in zip(self.split_times, self.split_predictions[r][c]):
    #                     ax.plot(time, self.split_fluxes[r][c] - chunk, '.')
    #             else:
    #                 merged_prediction = np.concatenate(self.split_predictions[r][c])
    #                 ax.plot(self.time, self.fluxes[r][c] - merged_prediction, label=f"({r}, {c})")
    #                 ax.legend()
    #     fig.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()
    


    # def lsq(
    #     self, cpm_reg, rescale=True, polynomials=False, poly_reg=None, target_fluxes=None, design_matrix=None
    # ):
    #     """Perform linear least squares with L2-regularization to find the coefficients for the model.

    #     .. note:: Although we have the flux errors, we chose not to include them (i.e. not do weighted least squares)
    #                 for computational efficiency for now. The errors are also not significantly different
    #                 across the entire measurement duration and are very small compared to the flux values.

    #     Args:
    #         cpm_reg (float): The L2-regularization value. Setting this argument to ``0`` removes
    #                     the regularization and is equivalent to performing ordinary least squares.
    #         rescale (Optional[boolean]): Choose whether to use zero-centered and median rescaled values
    #                     when performing least squares. The default is ``True`` and is recommended for numerical stability.
    #         polynomials (Optional[boolean]): Choose whether to include a set of polynomials (1, t, t^2, t^3)
    #                     as model components. The default is ``False``.
    #         poly_reg (Optional[float]): The L2-regularization value for the polynomial model.
    #         y (Optional[array]): Manually pass the target fluxes to use
    #         m (Optional[array]): Manually pass the design matrix to use

    #     Returns:
    #         List: Contains the lightcurve predicted by this method and also the
    #             difference light curve (data - model). The difference lightcurve is background-corrected light curve.
    #     """
    #     if (
    #         (self.is_target_set == False)
    #         or (self.is_exclusion_set == False)
    #         or (self.are_predictors_set == False)
    #     ):
    #         print("You missed a step.")

    #     self.cpm_reg = cpm_reg
    #     num_components = self.num_predictor_pixels
    #     self.fit_rescale = rescale
    #     self.fit_polynomials = polynomials

    #     reg_matrix = cpm_reg * np.identity(num_components)

    #     if (target_fluxes is None) & (design_matrix is None):
    #         if rescale == False:
    #             print("Calculating parameters using unscaled values.")
    #             y = self.target_fluxes
    #             m = (self.predictor_pixels_fluxes)  # Shape is (num of measurements, num of predictors)

    #         elif rescale == True:
    #             y = self.centered_scaled_target_fluxes
    #             m = self.centered_scaled_predictor_pixels_fluxes
    #     else:
    #         y = target_fluxes
    #         m = design_matrix

    #     if polynomials:
    #         if self.poly_terms == None:
    #             print("You need to specify the polynomial model with the set_poly_model.")
    #             return
    #         if design_matrix is None:
    #             m = np.hstack((m, self.v_matrix))
    #         # print("Final Design Matrix Shape: {}".format(m.shape))
    #         # num_components = num_components + self.v_matrix.shape[1]
    #         num_components = num_components + self.poly_terms
    #         reg_matrix = np.hstack(
    #             (
    #                 np.repeat(cpm_reg, self.num_predictor_pixels),
    #                 np.repeat(self.poly_reg, self.poly_terms),
    #             )
    #         ) * np.identity(num_components)

    #     # If it's the first time being called, store the original/full design matrix
    #     if (self.trained == False):
    #         self.full_m = m
    #     self.m = m

    #     # l = reg*np.identity(num_components)
    #     a = np.dot(m.T, m) + reg_matrix
    #     b = np.dot(m.T, y)

    #     self.all_params = np.linalg.solve(a, b)
    #     self.cpm_params = self.all_params[: self.num_predictor_pixels]
    #     self.poly_params = self.all_params[self.num_predictor_pixels :]

    #     self.all_prediction = np.dot(self.full_m, self.all_params)
    #     if polynomials:
    #         self.const_prediction = self.poly_params[0]  # Constant offset
    #         self.cpm_prediction = np.dot(self.full_m[:, : self.num_predictor_pixels], self.cpm_params)
    #         self.poly_prediction = (np.dot(self.full_m[:, self.num_predictor_pixels :], self.poly_params)- self.const_prediction
    #         )

    #     self.trained = True
    #     if rescale:
    #         self.difference = self.centered_scaled_target_fluxes - self.all_prediction
    #     else:
    #         self.difference = self.target_fluxes - self.all_prediction
    #     return [self.all_prediction, self.difference]

    # def get_hyperparameters(self, rescale=True, polynomials=False, k=10, grid_size=30, transit_duration=13):
    #     """Obtain the regularization hyperparameters by performing cross validation.

    #     """
    #     if (
    #         (self.is_target_set == False)
    #         or (self.is_exclusion_set == False)
    #         or (self.are_predictors_set == False)
    #     ):
    #         print("You missed a step.")
    #     print(f"Performing {k}-fold cross validation.")

    #     if rescale:
    #         y = self.centered_scaled_target_fluxes
    #         m = self.centered_scaled_predictor_pixels_fluxes
    #     else:
    #         y = self.target_fluxes
    #         m = self.predictor_pixels_fluxes

    #     num_components = self.num_predictor_pixels

    #     if polynomials:
    #         if self.poly_terms == None:
    #             print("You need to specify the polynomial model with the set_poly_model.")
    #             return
    #         m = np.hstack((m, self.v_matrix))
    #         num_components += self.poly_terms
    #         # reg_matrix = np.hstack(
    #         #     (
    #         #         np.repeat(cpm_reg, self.num_predictor_pixels),
    #         #         np.repeat(self.poly_reg, self.poly_terms),
    #         #     )
    #         # ) * np.identity(num_components)

    #     kf = KFold(k)
    #     f, axes = plt.subplots(2, 1, figsize=(12, 8))
    #     counter = 0
    #     cdpps = np.zeros((k, grid_size))
    #     for train, test in kf.split(self.time):
    #         print(f"{counter+1}/{k}")
    #         y_train = y[train]
    #         m_train = m[train, :]

    #         y_test = y[test]
    #         m_test = m[test, :]

    #         testset_cdpp = []
    #         if rescale:
    #             cpm_regs = np.linspace(0.001, 0.2, grid_size)
    #         else:
    #             cpm_regs = np.linspace(0, 1e8, grid_size)
    #         if polynomials:
    #             poly_regs = cpm_regs

    #         for cpm_reg in cpm_regs:
    #             reg_matrix = cpm_reg * np.identity(num_components)
    #             a = np.dot(m_train.T, m_train) + reg_matrix
    #             b = np.dot(m_train.T, y_train)
    #             params = np.linalg.solve(a, b)
    #             prediction = np.dot(m_test, params)
    #             lc = lk.TessLightCurve(self.time[test], y_test - prediction)
    #             testset_cdpp.append(lc.estimate_cdpp(transit_duration=transit_duration))
    #         axes[0].plot(cpm_regs, testset_cdpp, ".-")
    #         axes[1].plot(lc.time, lc.flux)
    #         cdpps[counter] = testset_cdpp
    #         counter += 1
    #     return (cpm_regs, cdpps)

    # def _lsq(self, y, m, reg_matrix, mask=None):
    #     if mask is not None:
    #         m = m[~mask]
    #         y = y[~mask]
    #     a = np.dot(m.T, m) + reg_matrix
    #     b = np.dot(m.T, y)
    #     w = np.linalg.solve(a, b)