import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.model_selection import KFold


from .utils import summary_plot
from .cutout_data import CutoutData


class CPM(object):
    """A Causal Pixel Model object

    Args:
        cutout_data (CutoutData): A CutoutData instance to obtain all the values from.
    """

    name = "CPM"

    def __init__(self, cutout_data):
        if isinstance(cutout_data, CutoutData):
            self.cutout_data = cutout_data
            self.time = cutout_data.time

        self.target_row = None
        self.target_col = None
        self.target_fluxes = None
        self.target_flux_errors = None
        self.target_flux_median = None
        self.normalized_target_fluxes = None
        self.normalized_target_flux_errors = None
        self.mask_target_pixel = None

        self.exclusion_size = None
        self.method_exclusion = None
        self.mask_excluded_pixels = None

        self.method_choose_predictor_pixels = None
        self.num_predictor_pixels = None
        self.locations_predictor_pixels = None
        self.mask_predictor_pixels = None
        self.predictor_pixels_fluxes = None
        self.normalized_predictor_pixels_fluxes = None

        self.reg = None
        self.reg_matrix = None
        self.m = None
        self.params = None
        self.prediction = None

        self.is_target_set = False
        self.is_exclusion_set = False
        self.are_predictors_set = False

    def set_target(self, target_row=None, target_col=None):
        """Set the target pixel by specifying the row and column.

        Args:
            target_row (int): The row position of the target pixel in the image.
            target_col (int): The column position of the target pixel in the image.
        """
        self.target_row = target_row
        self.target_col = target_col
        self.target_fluxes = self.cutout_data.fluxes[:, target_row, target_col]
        self.target_errors = self.cutout_data.flux_errors[:, target_row, target_col]
        self.target_median = self.cutout_data.flux_medians[target_row, target_col]
        self.normalized_target_fluxes = self.cutout_data.normalized_fluxes[
            :, target_row, target_col
        ]
        self.normalized_target_flux_errors = self.cutout_data.normalized_flux_errors[
            :, target_row, target_col
        ]

        mask = np.full(self.cutout_data.fluxes[0].shape, False)
        mask[target_row, target_col] = 1
        self.mask_target_pixel = mask
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
        sidelength_x = self.cutout_data.cutout_sidelength_x
        sidelength_y = self.cutout_data.cutout_sidelength_y
        

        excluded_pixels = np.full(self.cutout_data.fluxes[0].shape, False)
        if method == "closest":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength_x),
                max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength_y),
            ] = True
        if method == "cross":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength_x), :
            ] = True
            excluded_pixels[
                :, max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength_y)
            ] = True
        if method == "row_exclude":
            excluded_pixels[
                max(0, r - exclusion_size) : min(r + exclusion_size + 1, sidelength_x), :
            ] = True
        if method == "col_exclude":
            excluded_pixels[
                :, max(0, c - exclusion_size) : min(c + exclusion_size + 1, sidelength_y)
            ] = True

        self.mask_excluded_pixels = excluded_pixels
        self.is_exclusion_set = True

    def set_predictor_pixels(self, n=256, method="similar_brightness", seed=None):
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
                    this method chooses the top ``n`` pixels with a similar trend to the target pixel.
                "random": Randomly choose ``n`` predictor pixels.
                "similar_brightness": Choose ``n`` predictor pixels based on how close a given pixel's median brightness 
                    is to the target pixel's median brightness. This method potentially chooses variable pixels which
                    is not ideal (default). 
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
        sidelength_x = self.cutout_data.cutout_sidelength_x
        sidelength_y = self.cutout_data.cutout_sidelength_y
        

        # I'm going to do this in 1D by assinging individual pixels a single index instead of two.
        coordinate_idx = np.arange(sidelength_x * sidelength_y)
        valid_idx = coordinate_idx[~self.mask_excluded_pixels.ravel()]
        # valid_idx = coordinate_idx[self.excluded_pixels_mask.mask.ravel()]

        if method == "cosine_similarity":
            valid_normalized_fluxes = self.cutout_data.flattened_normalized_fluxes[
                :, ~self.mask_excluded_pixels.ravel()
            ]
            cos_sim = np.dot(
                valid_normalized_fluxes.T, self.normalized_target_fluxes
            ) / (
                np.linalg.norm(valid_normalized_fluxes.T, axis=1)
                * np.linalg.norm(self.normalized_target_fluxes)
            )
            chosen_idx = valid_idx[np.argsort(cos_sim)[::-1][0:n]]

        if method == "random":
            chosen_idx = np.random.choice(valid_idx, size=n, replace=False)

        if method == "similar_brightness":
            valid_flux_medians = self.cutout_data.flattened_flux_medians[
                ~self.mask_excluded_pixels.ravel()
            ]
            diff = np.abs(valid_flux_medians - self.target_median)
            chosen_idx = valid_idx[np.argsort(diff)[0:n]]

        self.locations_predictor_pixels = np.array(
            [[idx // sidelength_y, idx % sidelength_y] for idx in chosen_idx]
        )
        loc = self.locations_predictor_pixels.T
        mask = np.full(self.cutout_data.fluxes[0].shape, False)
        mask[loc[0], loc[1]] = True  # pylint: disable=unsubscriptable-object
        self.predictor_pixels_fluxes = self.cutout_data.fluxes[:, loc[0], loc[1]]  # pylint: disable=unsubscriptable-object
        self.normalized_predictor_pixels_fluxes = self.cutout_data.normalized_fluxes[
            :, loc[0], loc[1]  # pylint: disable=unsubscriptable-object
        ]
        self.mask_predictor_pixels = mask
        self.are_predictors_set = True
        self.m = self.normalized_predictor_pixels_fluxes

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

    def set_L2_reg(self, reg):
        """Set the L2-regularization for the CPM model and generates the regularization matrix.

        Args:
            reg (float): The L2-regularization value.

        """
        self.reg = reg
        self.reg_matrix = reg * np.identity(self.num_predictor_pixels)

    def predict(self, m=None, params=None, mask=None):
        """Make a prediction for the CPM model.

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

    def plot_model(self, size_predictors=10):

        fig, ax = plt.subplots()
        self._plot_model_onto_axes(ax, size_predictors=size_predictors)
        plt.show()
        return fig, ax

    def _plot_model_onto_axes(self, ax, size_predictors=10):

        median_image = self.cutout_data.flux_medians
        ax.imshow(median_image, origin="lower", 
            vmin=np.nanpercentile(median_image, 10),
            vmax=np.nanpercentile(median_image, 90)
            )
        ax.imshow(np.ma.masked_where(self.mask_excluded_pixels==False, self.mask_excluded_pixels), origin="lower", cmap="Set1", alpha=0.5)
        # ax.imshow(np.ma.masked_where(self.mask_target_pixel==False, self.mask_target_pixel), origin="lower", cmap="binary", alpha=1.0)
        ax.scatter(self.target_col, self.target_row, marker="*", color="w", s=15, alpha=1.0)
        # ax.imshow(np.ma.masked_where(self.mask_predictor_pixels==False, self.mask_predictor_pixels), origin="lower", cmap="Set1", alpha=0.9)
        predictor_locs = self.locations_predictor_pixels.T 
        ax.scatter(predictor_locs[1], predictor_locs[0], marker="o", color="C3", s=size_predictors, alpha=0.9)  # pylint: disable=unsubscriptable-object
