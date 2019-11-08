import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.model_selection import KFold

from .utils import summary_plot
from .target_data import TargetData


class CPM(object):
    """A Causal Pixel Model object

    Args:
        target_data (TargetData): 
    """

    def __init__(self, target_data):
        if target_data is TargetData:
            self.target_data = target_data
        
        self.target_row = None
        self.target_col = None
        self.target_fluxes = None
        self.target_flux_errors = None
        self.target_flux_median = None
        self.centered_scaled_target_fluxes = None
        self.centered_scaled_target_flux_errors = None

        self.exclusion_size = None
        self.method_exclusion = None
        self.mask_excluded_pixels = None

        self.method_choose_predictor_pixels = None
        self.num_predictor_pixels = None
        self.locations_predictor_pixels = None
        self.mask_predictor_pixels = None
        self.predictor_pixels_fluxes = None
        self.centered_scaled_predictor_pixels_fluxes = None

        self.interval = None
        self.scaled_interval = None
        self.poly_terms = None
        self.v_matrix = None
        self.poly_reg = None

        self.cpm_reg = None
        self.fit_rescale = None
        self.fit_polynomials = None
        self.full_m = None
        self.m = None

        self.all_params = None
        self.cpm_params = None
        self.poly_params = None

        self.all_prediction = None
        self.const_prediction = None
        self.cpm_prediction = None
        self.poly_prediction = None

    def set_target(self, target_row=None, target_col=None):
        """Set the target pixel by specifying the row and column.

        Args:
            target_row (int): The row position of the target pixel in the image.
            target_col (int): The column position of the target pixel in the image.
        """
        if (target_row is None) and (target_col is None):
            target_row = self.target_data.cutout_sidelength / 2
            target_col = self.target_data.cutout_sidelength / 2
        else:
            self.target_row = target_row
            self.target_col = target_col
        self.target_fluxes = self.target_data.fluxes[:, target_row, target_col]
        self.target_errors = self.target_data.flux_errors[:, target_row, target_col]
        self.target_median = self.target_data.flux_medians[target_row, target_col]
        self.centered_scaled_target_fluxes = self.target_data.centered_scaled_target_fluxes[:, target_row, target_col]
        self.centered_scaled_target_flux_errors = self.target_data.centered_scaled_target_flux_errors[:, target_row, target_col]


        mask = np.zeros(self.target_data.fluxes[0].shape)
        mask[target_row, target_col] = 1
        self.target_pixel_mask = mask
        self.is_target_set = True

    def set_exclusion(self, exclusion_size=5, method="closest"):
        """Set the exclusion region around the target pixel. 
            The exclusion region is necessary to ensure that pixels from the same source are not chosen as predictor pixels.

        Args:
            exclusion_size (Optional[int]): Specifies the size of the exclusion region. The default setting is to exclude a square region
                with a sidelength of (2 * ``exclusion_size`` + 1) pixels surrounding the target pixel.
            method (Optional): Specify the method for excluding a region.
                "closest": Excludes a square region surrounding the target pixel (default).
                "cross": Excludes a cross region.
                "row_exclude": Excludes a set of rows.
                "col_exclude": Excludes a set of columns.
        """

        if self.is_target_set == False:
            print(
                "Please set the target pixel to predict using the set_target() method."
            )
            return

        r = self.target_row  # just to reduce verbosity for this function
        c = self.target_col
        self.exclusion_size = exclusion_size
        sidelength = self.target_data.cutout_sidelength

        excluded_pixels = np.zeros(self.target_data.fluxes[0].shape)
        if method == "closest":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength),
                max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength),
            ] = 1
        if method == "cross":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength), :
            ] = 1
            excluded_pixels[
                :, max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength)
            ] = 1
        if method == "row_exclude":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength), :
            ] = 1
        if method == "col_exclude":
            excluded_pixels[
                :, max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength)
            ] = 1

        self.mask_excluded_pixels = excluded_pixels
        self.is_exclusion_set = True

    def set_predictor_pixels(
        self, n=256, method="cosine_similarity", seed=None
    ):
        """Set the predictor pixels (features) used to perform CPM.
            CPM attempts to fit to the target pixel's light curve using the linear combination
            of the predictor pixels' light curves. The exclusion region must be set prior to 
            setting the predictor pixels as we do not want to use pixels close to the target pixel.

        Args:
            n (Optional[int]): Number of predictor pixels to use. This number should change based on 
                the size of your cutout (the number of pixels in your image). 
                The default setting is 256 (for a 100x100 pixel cutout). If your cutout is smaller 
                make this number smaller and if your cutout is larger you might get a better fit if you
                increase this number.
            method (Optional): Specify the method for choosing predictor pixels. Pixels in the excluded region are never chosen.
                "cosine_similarity": Choose ``n`` predictor pixels based on the cosine similarity of a given 
                    pixel's light curve with the target pixel's lightcurve. In other words,
                    this method chooses the top ``n`` pixels with a similar trend to the target pixel (default).
                "random": Randomly choose ``n`` predictor pixels.
                "similar_brightness": Choose ``n`` predictor pixels based on how close a given pixel's median brightness 
                    is to the target pixel's median brightness. This method potentially chooses variable pixels which
                    probably is not ideal. 
            seed (Optional[int]): The seed passed to ``np.random.seed`` to be able to reproduce predictor pixels using 
                the "random" method. The other methods are deterministic and are always reproducible.
        """
        
        if seed != None:
            np.random.seed(seed=seed)

        if (self.is_target_set == False) or (self.is_exclusion_set == False):
            print("Please set the target pixel and exclusion region.")
            return

        self.method_choose_predictor_pixels = method
        self.num_predictor_pixels = n
        sidelength = self.target_data.fluxes.shape[1]

        # I'm going to do this in 1D by assinging individual pixels a single index instead of two.
        coordinate_idx = np.arange(sidelength ** 2)
        valid_idx = coordinate_idx[~self.mask_excluded_pixels.ravel()]
        # valid_idx = coordinate_idx[self.excluded_pixels_mask.mask.ravel()]

        if method == "cosine_similarity":
            valid_centered_scaled_fluxes = self.flattened_centered_scaled_fluxes[
                :, self.mask_excluded_pixels.ravel()]
            cos_sim = np.dot(
                valid_centered_scaled_fluxes.T, self.centered_scaled_target_fluxes
            ) / (
                np.linalg.norm(valid_centered_scaled_fluxes.T, axis=1)
                * np.linalg.norm(self.centered_scaled_target_fluxes)
            )
            chosen_idx = valid_idx[
                np.argsort(cos_sim)[::-1][0 : n]
            ]

        if method == "random":
            chosen_idx = np.random.choice(
                valid_idx, size=n, replace=False
            )

        if method == "similar_brightness":
            valid_flux_medians = self.target_data.flattened_flux_medians[
                self.mask_excluded_pixels.ravel()]
            diff = np.abs(valid_flux_medians - self.target_median)
            chosen_idx = valid_idx[np.argsort(diff)[0 : n]]

        self.locations_predictor_pixels = np.array(
            [[idx // sidelength, idx % sidelength] for idx in chosen_idx]
        )
        loc = self.locations_predictor_pixels.T
        mask = np.zeros(self.target_data.fluxes[0].shape)
        mask[loc[0], loc[1]] = 1
        self.predictor_pixels_fluxes = self.target_data.fluxes[:, loc[0], loc[1]]
        self.centered_scaled_predictor_pixels_fluxes = self.target_data.fluxes[
            :, loc[0], loc[1]
        ]
        self.mask_predictor_pixels = mask
        self.are_predictors_set = True

    def set_target_exclusion_predictors(
        self,
        target_row,
        target_col,
        exclusion_size=5,
        exclusion_method="closest",
        n=256,
        predictor_method="cosine_similarity",
        seed=None,
    ):
        """Convenience function that simply calls the set_target(), set_exclusion(), set_predictor_pixels() functions sequentially
        """
        self.set_target(target_row, target_col)
        self.set_exclusion(exclusion_size, method=exclusion_method)
        self.set_predictor_pixels(n, method=predictor_method, seed=seed)

    def set_poly_model(self, interval=2, poly_terms=4):
        """Set the polynomial model parameters. The polynomial model is used to capture long term trends in the data
            believed to be signal and not background noise (such as supernova lightcurves). This method is essentially 
            calling the ``numpy.vander()`` method.

        Args:
            interval (Optional[int]): Sets the min and max value of the input vector to pass to ``numpy.vander``.
                The larger this value, the more flexibility the polynomial model will have for a given number of powers.
            poly_terms (Optional[int]): Specify the number of "powers" to use in the polynomial model.
                As the first power is a constant, the highest power is ``poly_terms - 1``. 

        """
        self.interval = interval
        x = self.target_data.time
        centered_x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min())
        self.scaled_interval = interval * centered_x
        self.poly_terms = poly_terms
        self.v_matrix = np.vander(self.scaled_interval, N=poly_terms, increasing=True)

    def set_poly_reg(self, poly_reg):
        self.poly_reg = poly_reg

    def lsq(
        self, cpm_reg, rescale=True, polynomials=False, poly_reg=None, target_fluxes=None, design_matrix=None
    ):
        """Perform linear least squares with L2-regularization to find the coefficients for the model.

        .. note:: Although we have the flux errors, we chose not to include them (i.e. not do weighted least squares)
                    for computational efficiency for now. The errors are also not significantly different
                    across the entire measurement duration and are very small compared to the flux values.

        Args:
            cpm_reg (float): The L2-regularization value. Setting this argument to ``0`` removes
                        the regularization and is equivalent to performing ordinary least squares.
            rescale (Optional[boolean]): Choose whether to use zero-centered and median rescaled values
                        when performing least squares. The default is ``True`` and is recommended for numerical stability.
            polynomials (Optional[boolean]): Choose whether to include a set of polynomials (1, t, t^2, t^3)
                        as model components. The default is ``False``.
            poly_reg (Optional[float]): The L2-regularization value for the polynomial model.
            y (Optional[array]): Manually pass the target fluxes to use
            m (Optional[array]): Manually pass the design matrix to use 
        """
        if (
            (self.is_target_set == False)
            or (self.is_exclusion_set == False)
            or (self.are_predictors_set == False)
        ):
            print("You missed a step.")

        self.cpm_reg = cpm_reg
        num_components = self.num_predictor_pixels
        self.fit_rescale = rescale
        self.fit_polynomials = polynomials

        reg_matrix = cpm_reg * np.identity(num_components)

        if (y is None) & (m is None):
            if rescale == False:
                print("Calculating parameters using unscaled values.")
                y = self.target_fluxes
                m = (self.predictor_pixels_fluxes)  # Shape is (num of measurements, num of predictors)

            elif rescale == True:
                y = self.centered_scaled_target_fluxes
                m = self.centered_scaled_predictor_pixels_fluxes
        else:
            y = target_fluxes
            m = design_matrix

        if polynomials:
            if self.poly_terms = None:
                print("You need to specify the polynomial model with the set_poly_model.")
                return
            if design_matrix is None:
                m = np.hstack((m, self.v_matrix))
            # print("Final Design Matrix Shape: {}".format(m.shape))
            # num_components = num_components + self.v_matrix.shape[1]
            num_components = num_components + self.poly_terms
            reg_matrix = np.hstack(
                (
                    np.repeat(cpm_reg, self.num_predictor_pixels),
                    np.repeat(self.poly_reg, self.poly_terms),
                )
            ) * np.identity(num_components)

        # If it's the first time being called, store the original/full design matrix
        if (self.trained == False):
            self.full_m = m
        self.m = m

        # l = reg*np.identity(num_components)
        a = np.dot(m.T, m) + reg_matrix
        b = np.dot(m.T, y)

        self.all_params = np.linalg.solve(a, b)
        self.cpm_params = self.all_params[: self.num_predictor_pixels]
        self.poly_params = self.all_params[self.num_predictor_pixels :]

        self.all_prediction = np.dot(self.full_m, self.lsq_params)
        if polynomials:
            self.const_prediction = self.poly_params[0]  # Constant offset
            self.cpm_prediction = np.dot(self.full_m[:, : self.num_predictor_pixels], self.cpm_params)
            self.poly_prediction = (np.dot(self.full_m[:, self.num_predictor_pixels :], self.poly_params)- self.const_prediction
            )

        # if (rescale == True):
        #     self.lsq_prediction = np.median(self.target_fluxes)*(self.lsq_prediction + 1)
        #     if (polynomials == True):
        #         self.constant_prediction = np.median(self.target_fluxes)*self.poly_params[0]
        #         self.cpm_prediction = np.median(self.target_fluxes)*(self.cpm_prediction + 1)
        #         self.poly_prediction = np.median(self.target_fluxes)*(self.poly_prediction + 1) - self.constant_prediction

        self.trained = True
        self.residual = self.rescaled_target_fluxes - self.lsq_prediction
        return (self.all_prediction, self.residual)


    # def __init__(self, fits_file, remove_bad=True, verbose=True):
    #     self.file_path = fits_file
    #     self.file_name = fits_file.split("/")[-1]
    #     with fits.open(fits_file, mode="readonly") as hdu:
    #         self.time = hdu[1].data["TIME"]
    #         self.im_fluxes = hdu[1].data["FLUX"]
    #         self.im_errors = hdu[1].data["FLUX_ERR"]
    #         self.quality = hdu[1].data["QUALITY"]
    #         try:
    #             self.wcs_info = WCS(hdulist[2].header)
    #         except:
    #             if verbose == True:
    #                 print("WCS Info could not be retrieved")

    #     self.dump_times = self.time[self.quality > 0]
    #     # If remove_bad is set to True, we'll remove the values with a nonzero entry in the quality array
    #     if remove_bad == True:
    #         bool_good = (
    #             self.quality == 0
    #         )  # The zero value entries for the quality array are the "good" values
    #         if verbose == True:
    #             print(
    #                 'Removing {} bad data points (out of {}) using the TESS provided "QUALITY" array'.format(
    #                     np.sum(~bool_good), np.size(bool_good)
    #                 )
    #             )
    #         self.time = self.time[bool_good]
    #         self.im_fluxes = self.im_fluxes[bool_good]
    #         self.im_errors = self.im_errors[bool_good]

    #     # We're going to precompute the pixel lightcurve medians since it's used to set the predictor pixels
    #     # but never has to be recomputed. nanmedian is used to handle images containing NaN values.
    #     self.pixel_medians = np.nanmedian(self.im_fluxes, axis=0)
    #     self.im_sidelength = self.im_fluxes[0].shape[0]
    #     self.flattened_pixel_medians = self.pixel_medians.reshape(
    #         self.im_sidelength ** 2
    #     )
    #     self.rescaled_im_fluxes = (self.im_fluxes / self.pixel_medians) - 1
    #     self.flattened_rescaled_im_fluxes = self.rescaled_im_fluxes.reshape(
    #         self.time.shape[0], self.im_sidelength ** 2
    #     )

        # self.target_row = None
        # self.target_col = None
        # self.target_fluxes = None
        # self.target_errors = None
        # self.target_median = None
        # self.rescaled_target_fluxes = None
        # self.rescaled_target_errors = None
        # self.target_pixel_mask = None

        # self.exclusion = None
        # self.excluded_pixels_mask = None

        # self.method_predictor_pixels = None
        # self.num_predictor_pixels = None
        # self.predictor_pixels_locations = None
        # self.predictor_pixels_mask = None
        # self.predictor_pixels_fluxes = None
        # self.rescaled_predictor_pixels_fluxes = None

        self.rescale = None
        self.polynomials = None
        self.cpm_regularization = None
        self.lsq_params = None
        self.cpm_params = None
        self.poly_params = None
        self.full_m = None
        self.m = None

        self.const_prediction = None
        self.cpm_prediction = None
        self.poly_prediction = None
        self.residual = None
        self.im_predicted_fluxes = None
        self.im_diff = None

        self.is_target_set = False
        self.is_exclusion_set = False
        self.are_predictors_set = False
        self.trained = False
        self.over_entire_image = False
        self.valid = None

        self.centered_time = None
        self.scaled_centered_time = None
        self.time_interval = None
        self.poly_terms = None
        self.v_matrix = None
        self.poly_reg = None

        self.lc_matrix = None
        self.aperture_lc = None
    

    def xval(self, cpm_reg, rescale=True, polynomials=False, k=10):
        if (
            (self.is_target_set == False)
            or (self.is_exclusion_set == False)
            or (self.are_predictors_set == False)
        ):
            print("You missed a step.")

        self.cpm_regularization = cpm_reg
        num_components = self.num_predictor_pixels
        self.rescale = rescale
        self.polynomials = polynomials
        reg_matrix = cpm_reg * np.identity(num_components)

        y = self.rescaled_target_fluxes
        m = self.rescaled_predictor_pixels_fluxes
        self.m = m

        if polynomials == True:
            m = np.hstack((m, self.v_matrix))
            # print("Final Design Matrix Shape: {}".format(m.shape))
            num_components = num_components + self.v_matrix.shape[1]
            reg_matrix = np.hstack(
                (
                    np.repeat(cpm_reg, self.num_predictor_pixels),
                    np.repeat(self.poly_reg, self.poly_terms),
                )
            ) * np.identity(num_components)

        prediction = []
        res = []
        kf = KFold(k)
        for train, test in kf.split(self.time):
            y_train = y[train]
            m_train = m[train, :]

            y_test = y[test]
            m_test = m[test, :]

            a = np.dot(m_train.T, m_train) + reg_matrix
            b = np.dot(m_train.T, y_train)

            self.lsq_params = np.linalg.solve(a, b)
            self.cpm_params = self.lsq_params[: self.num_predictor_pixels]
            self.poly_params = self.lsq_params[self.num_predictor_pixels :]

            self.lsq_prediction = np.dot(m_test, self.lsq_params)
            self.const_prediction = None
            self.cpm_prediction = None
            self.poly_prediction = None

            if polynomials == True:
                self.const_prediction = self.poly_params[0]  # Constant offset
                self.cpm_prediction = np.dot(
                    m_test[:, : self.num_predictor_pixels], self.cpm_params
                )
                self.poly_prediction = (
                    np.dot(m_test[:, self.num_predictor_pixels :], self.poly_params)
                    - self.const_prediction
                )

            self.trained = True
            prediction.append(self.lsq_prediction)
            res.append(self.lsq_prediction)
            self.residual = y_test - self.lsq_prediction
            plt.plot(self.time[test], self.residual, ".-")

        return (prediction, res)

    # def lsq(
    #     self, cpm_reg, rescale=True, polynomials=False, updated_y=None, updated_m=None
    # ):
    #     """Perform linear least squares with L2-regularization to find the coefficients for the model.

    #     .. note:: Although we have the flux errors, we chose not to include them (i.e. not do weighted least squares)
    #                 for computational efficiency for now. The errors are also not significantly different
    #                 across the entire measurement duration and are very small compared to the flux values.

    #     Args:
    #         cpm_reg (int): The L2-regularization value. Setting this argument to ``0`` removes
    #                     the regularization and is equivalent to performing ordinary least squares.
    #         rescale (Optional[boolean]): Choose whether to use zero-centered and median rescaled values
    #                     when performing least squares. The default is ``True`` and is recommended for numerical stability.
    #         polynomials (Optional[boolean]): Choose whether to include a set of polynomials (1, t, t^2, t^3)
    #                     as model components. The default is ``False``.
    #         updated_y (Optional[array]): Manually pass the target fluxes to use
    #         updated_m (Optionam[array]): Manually pass the design matrix to use 
    #     """
    #     if (
    #         (self.is_target_set == False)
    #         or (self.is_exclusion_set == False)
    #         or (self.are_predictors_set == False)
    #     ):
    #         print("You missed a step.")
    #     self.cpm_regularization = cpm_reg
    #     num_components = self.num_predictor_pixels
    #     self.rescale = rescale
    #     self.polynomials = polynomials
    #     reg_matrix = cpm_reg * np.identity(num_components)

    #     if (updated_y is None) & (updated_m is None):
    #         if rescale == False:
    #             print("Calculating parameters using unscaled values.")
    #             y = self.target_fluxes
    #             m = (
    #                 self.predictor_pixels_fluxes
    #             )  # Shape is (num of measurements, num of predictors)

    #         elif rescale == True:
    #             y = self.rescaled_target_fluxes
    #             m = self.rescaled_predictor_pixels_fluxes
    #     else:
    #         y = updated_y
    #         m = updated_m

    #     # # This is such a hack I need to fix this (August 2nd, 2019)
    #     # if reg_matrix is None:
    #     #     reg_matrix = cpm_reg*np.identity(num_components)

    #     if polynomials == True:
    #         if updated_m is None:
    #             m = np.hstack((m, self.v_matrix))
    #         # print("Final Design Matrix Shape: {}".format(m.shape))
    #         num_components = num_components + self.v_matrix.shape[1]
    #         reg_matrix = np.hstack(
    #             (
    #                 np.repeat(cpm_reg, self.num_predictor_pixels),
    #                 np.repeat(self.poly_reg, self.poly_terms),
    #             )
    #         ) * np.identity(num_components)

    #     if (
    #         self.trained == False
    #     ):  # if it's the first time being called, store the original/full design matrix
    #         self.full_m = m
    #     self.m = m

    #     # l = reg*np.identity(num_components)
    #     a = np.dot(m.T, m) + reg_matrix
    #     b = np.dot(m.T, y)

    #     self.lsq_params = np.linalg.solve(a, b)
    #     self.cpm_params = self.lsq_params[: self.num_predictor_pixels]
    #     self.poly_params = self.lsq_params[self.num_predictor_pixels :]
    #     # self.lsq_prediction = np.dot(m, self.lsq_params)

    #     self.lsq_prediction = np.dot(self.full_m, self.lsq_params)
    #     self.const_prediction = None
    #     self.cpm_prediction = None
    #     self.poly_prediction = None

    #     if polynomials == True:
    #         self.const_prediction = self.poly_params[0]  # Constant offset
    #         # self.cpm_prediction = np.dot(m[:, :self.num_predictor_pixels], self.cpm_params)
    #         # self.poly_prediction = np.dot(m[:, self.num_predictor_pixels:], self.poly_params) - self.const_prediction

    #         self.cpm_prediction = np.dot(
    #             self.full_m[:, : self.num_predictor_pixels], self.cpm_params
    #         )
    #         self.poly_prediction = (
    #             np.dot(self.full_m[:, self.num_predictor_pixels :], self.poly_params)
    #             - self.const_prediction
    #         )

    #     # if (rescale == True):
    #     #     self.lsq_prediction = np.median(self.target_fluxes)*(self.lsq_prediction + 1)
    #     #     if (polynomials == True):
    #     #         self.constant_prediction = np.median(self.target_fluxes)*self.poly_params[0]
    #     #         self.cpm_prediction = np.median(self.target_fluxes)*(self.cpm_prediction + 1)
    #     #         self.poly_prediction = np.median(self.target_fluxes)*(self.poly_prediction + 1) - self.constant_prediction

    #     self.trained = True
    #     self.residual = self.rescaled_target_fluxes - self.lsq_prediction
    #     return (self.lsq_prediction, self.residual)

    def get_contributing_pixels(self, number):
        """Return the n-most contributing pixels' locations and a mask to see them
        """
        if self.trained == False:
            print("You need to train the model first.")

        idx = np.argsort(np.abs(self.cpm_params))[: -(number + 1) : -1]
        top_n_loc = self.predictor_pixels_locations[idx]
        loc = top_n_loc.T
        top_n = np.zeros(self.im_fluxes[0].shape)
        top_n[loc[0], loc[1]] = 1
        top_n_mask = np.ma.masked_where(top_n == 0, top_n)

        return (top_n_loc, top_n_mask)

    def get_aperture_lc(
        self, rows=None, cols=None, box=1, show_pixel_lc=False, show_aperture_lc=False
    ):
        if (rows is None) & (cols is None):
            t_row = self.target_row
            t_col = self.target_col
            rows = np.arange(t_row - box, t_row + box + 1)[::-1]
            cols = np.arange(t_col - box, t_col + box + 1, 1)

        # rows = rows[::-1]
        self.lc_matrix = np.zeros((rows.size, cols.size, self.time.size))
        for row in rows:
            for col in cols:
                pix_cpm = CPM(self.file_path, remove_bad=True, verbose=False)
                pix_cpm.set_target(row, col)
                pix_cpm.set_exclusion(10, method="closest")
                pix_cpm.set_predictor_pixels(256, method="cosine_similarity")
                prediction, corrected_lc = pix_cpm.lsq(self.cpm_regularization)
                self.lc_matrix[np.abs(row - rows[0]), col - cols[0]] = corrected_lc

        self.aperture_lc = np.sum(self.lc_matrix, axis=(0, 1))

        if show_pixel_lc == True:
            fig, axs = plt.subplots(
                rows.size, cols.size, sharex=True, sharey=True, figsize=(14, 8)
            )
            row_idx = np.arange(rows.size)
            col_idx = np.arange(cols.size)
            for row in row_idx:
                for col in col_idx:
                    ax = axs[row, col]
                    ax.plot(self.time, self.lc_matrix[row, col], ".", color="black")
                    # plt.show()
            fig.subplots_adjust(wspace=0, hspace=0)

        if show_aperture_lc == True:
            plt.figure(figsize=(14, 8))
            plt.plot(self.time, self.aperture_lc, ".", color="black")
        return self.aperture_lc, self.lc_matrix

    # def _batch(
    #     self,
    #     cpm_reg,
    #     rows,
    #     cols,
    #     exclusion=4,
    #     exclusion_method="closest",
    #     num_predictor_pixels=256,
    #     predictor_method="similar_brightness",
    #     rescale=True,
    #     polynomials=False,
    # ):
    #     self.cpm_regularization = cpm_reg
    #     self.im_predicted_fluxes = np.empty(self.im_fluxes.shape)
    #     for (row, col) in zip(rows, cols):
    #         self.set_target(row, col)
    #         self.set_exclusion(exclusion, method=exclusion_method)
    #         self.set_predictor_pixels(num_predictor_pixels, method=predictor_method)
    #         self.lsq(cpm_reg, rescale=rescale, polynomials=polynomials)
    #         if polynomials == True:
    #             self.im_predicted_fluxes[:, row, col] = (
    #                 self.cpm_prediction + self.const_prediction
    #             )
    #         elif polynomials == False:
    #             self.im_predicted_fluxes[:, row, col] = self.lsq_prediction
    #     self.im_diff = self.rescaled_im_fluxes - self.im_predicted_fluxes

    # def entire_image(
    #     self,
    #     cpm_reg,
    #     exclusion=4,
    #     exclusion_method="closest",
    #     num_predictor_pixels=256,
    #     predictor_method="similar_brightness",
    #     rescale=True,
    #     polynomials=False,
    # ):
    #     num_col = self.im_fluxes[0].shape[1]
    #     idx = np.arange(num_col ** 2)
    #     rows = idx // num_col
    #     cols = idx % num_col

    #     self._batch(
    #         cpm_reg,
    #         rows,
    #         cols,
    #         exclusion=exclusion,
    #         exclusion_method=exclusion_method,
    #         num_predictor_pixels=num_predictor_pixels,
    #         predictor_method=predictor_method,
    #         rescale=rescale,
    #         polynomials=polynomials,
    #     )

    #     self.over_entire_image = True

    # def difference_image_sap(
    #     self,
    #     cpm_reg,
    #     row,
    #     col,
    #     size,
    #     exclusion=10,
    #     exclusion_method="closest",
    #     num_predictor_pixels=256,
    #     predictor_method="similar_brightness",
    #     rescale=True,
    #     polynomials=True,
    # ):
    #     """Simple Aperture Photometry for a given pixel in the difference images
    #     """

    #     if self.over_entire_image == False:
    #         side = 2 * size + 1

    #         rows = np.repeat(np.arange(row - size, row + size + 1), side)
    #         cols = np.tile(np.arange(col - size, col + size + 1), side)

    #         self._batch(
    #             cpm_reg,
    #             rows,
    #             cols,
    #             exclusion=exclusion,
    #             exclusion_method=exclusion_method,
    #             num_predictor_pixels=num_predictor_pixels,
    #             predictor_method=predictor_method,
    #             rescale=rescale,
    #             polynomials=polynomials,
    #         )

    #     aperture = self.im_diff[
    #         :,
    #         max(0, row - size) : min(row + size + 1, self.im_diff.shape[1]),
    #         max(0, col - size) : min(col + size + 1, self.im_diff.shape[1]),
    #     ]
    #     aperture_lc = np.sum(aperture, axis=(1, 2))
    #     return aperture, aperture_lc

    def sigma_clip_process(self, sigma=5, subtract_polynomials=False):

        valid = np.full(self.time.shape[0], True)
        total_clipped_counter = 0
        prev_clipped_counter = 0
        iter_num = 1
        while True:
            if (subtract_polynomials == False) & (self.cpm_prediction is not None):
                model = self.cpm_prediction + self.const_prediction
            else:
                model = self.lsq_prediction

            diff = self.rescaled_target_fluxes - model
            # print(np.sum(valid))
            # print(diff[valid].shape[0])
            # sigma_boundary = sigma*np.sqrt(np.sum(np.abs(diff[valid])**2) / np.sum(valid))
            sigma_boundary = sigma * np.sqrt(np.sum((diff[valid]) ** 2) / np.sum(valid))
            # print(sigma_boundary)
            valid[np.abs(diff) > sigma_boundary] = False
            total_clipped_counter = np.sum(~valid)
            current_clipped_counter = total_clipped_counter - prev_clipped_counter
            if current_clipped_counter == 0:
                break
            print(
                "Iteration {}: Removing {} data points".format(
                    iter_num, current_clipped_counter
                )
            )
            prev_clipped_counter += current_clipped_counter
            iter_num += 1
            self.valid = valid
            self._rerun()
            # post_par = self.lsq_params
            # print("This better be false: {}".format(np.all(pre_par == post_par)))

    def _rerun(self):
        updated_y = self.rescaled_target_fluxes[self.valid]
        updated_m = self.full_m[self.valid, :]
        self.lsq(
            self.cpm_regularization,
            self.rescale,
            self.polynomials,
            updated_y,
            updated_m,
        )

    def _reset(self):
        self.__init__(self.file_path)
