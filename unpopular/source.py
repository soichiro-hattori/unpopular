import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import lightkurve as lk
from scipy.ndimage import median_filter
from matplotlib.ticker import MaxNLocator

from .cutout_data import CutoutData
from .model import PixelModel
from .cpm_model import CPM
from .poly_model import PolyModel


class Source(object):
    """The main interface to interact with both the data and models for a TESS source

    """

    def __init__(self, path, remove_bad=True, verbose=True, 
                 provenance='TessCut', quality=None, bkg_subtract=False, bkg_n=100,
                 time_path=None, flux_path=None, ferr_path=None):
        self.provenance = provenance
        if provenance in ['TessCut', 'eleanor']:
            self.cutout_data = CutoutData(path, remove_bad, verbose, 
                                        self.provenance, quality, bkg_subtract, bkg_n)
        elif provenance == 'local':
            self.cutout_data = CutoutData(path, remove_bad, verbose,
                                         self.provenance, bkg_subtract=bkg_subtract, bkg_n=bkg_n,
                                         time_path=time_path, flux_path=flux_path, ferr_path=ferr_path)
        self.time = self.cutout_data.time
        self.aperture = None
        self.models = None
        self.fluxes = None
        self.flux_errs = None
        self.predictions = None
        self.detrended_lcs = None
        self.split_times = None
        self.split_predictions = None
        self.split_fluxes = None
        self.split_detrended_lcs = None


    def set_aperture(self, rowlims=[49, 51], collims=[49, 51]):
        self.models = []
        self.fluxes = []
        self.flux_errs = []
        apt = np.full(self.cutout_data.fluxes[0].shape, False)
        # print("Assuming you're interested in the central set of pixels")
        apt[rowlims[0]:rowlims[1]+1, collims[0]:collims[1]+1] = True

        self.aperture = apt
        for row in range(rowlims[0], rowlims[1]+1):
            row_models = []
            row_fluxes = []
            row_ferrs = []
            for col in range(collims[0], collims[1]+1):
                row_models.append(PixelModel(self.cutout_data, row, col))
                row_fluxes.append(self.cutout_data.normalized_fluxes[:, row, col])
                row_ferrs.append(self.cutout_data.normalized_flux_errors[:, row, col])
            self.models.append(row_models)
            self.fluxes.append(row_fluxes)
            self.flux_errs.append(row_ferrs)

    def set_aperture_via_mask(self, mask):
        data_shape = self.cutout_data.fluxes[0].shape
        try:
            assert mask.shape == data_shape
        except AssertionError:
            print("The mask and FFI cutout must have the same shape.")
            print(f"mask shape: {mask.shape}, cutout shape: {data_shape}")
        self.models = []
        self.fluxes = []
        self.flux_errs = []

        self.aperture = mask
        rows, cols = np.asarray(mask == True).nonzero()
        for r, c in zip(rows, cols):
            _models, _fluxes, _flux_errs = [], [], []
            _models.append(PixelModel(self.cutout_data, r, c))
            self.models.append(_models)
            _fluxes.append(self.cutout_data.normalized_fluxes[:, r, c])
            self.fluxes.append(_fluxes)
            _flux_errs.append(self.cutout_data.normalized_flux_errors[:,r,c])
            self.flux_errs.append(_flux_errs)


    def add_cpm_model(self, exclusion_size=5,
        exclusion_method="closest",
        n=256,
        predictor_method="similar_brightness",
        seed=None):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.add_cpm_model(exclusion_size, exclusion_method, n, predictor_method, seed)

    def remove_cpm_model(self):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.remove_cpm_model()

    def add_poly_model(self, scale=2, num_terms=4):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.add_poly_model(scale, num_terms)

    def remove_poly_model(self):
        if self.models is None:
            print("Please set the aperture first.")
        for row_models in self.models:
            for model in row_models:
                model.remove_poly_model()

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

    def holdout_fit_predict(self, k=10, mask=None, verbose=False):
        if self.models is None:
            print("Please set the aperture first.")
        if mask is not None:
            print(f"Using user-provided mask. Clipping {np.sum(~mask)} points.")  # pylint: disable=invalid-unary-operand-type 
        predictions = []
        fluxes = []
        detrended_lcs = []
        for row_models in self.models:
            row_predictions = []
            row_fluxes = []
            # row_detrended_lcs = []
            for model in row_models:
                times, flux, pred = model.holdout_fit_predict(k, mask, verbose=verbose)
                row_fluxes.append(flux)
                row_predictions.append(pred)
                # row_detrended_lcs.append(flux - pred)
            fluxes.append(row_fluxes)
            predictions.append(row_predictions)
            # detrended_lcs.append(row_detrended_lcs)
        self.split_times = times
        self.split_fluxes = fluxes
        self.split_predictions = predictions
        self.split_detrended_lcs = detrended_lcs
        self.rescale()
        return (times, fluxes, predictions)

    def plot_cutout(self, rowlims=None, collims=None, l=10, h=90, show_aperture=False, projection=None):
        if rowlims is None:
            rows = [0, self.cutout_data.cutout_sidelength_x]
        else:
            rows = rowlims

        if collims is None:
            cols = [0, self.cutout_data.cutout_sidelength_y]
        else:
            cols = collims
        full_median_image = self.cutout_data.flux_medians
        median_image = self.cutout_data.flux_medians[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
        if projection == "wcs":
            projection = self.cutout_data.wcs_info
        plt.subplot(111, projection=projection)
        plt.imshow(
            median_image,
            origin="lower",
            vmin=np.percentile(full_median_image, l),
            vmax=np.percentile(full_median_image, h),
        )
        if rowlims is not None:
            plt.yticks(np.arange(rowlims[-1]+1-rowlims[0]), labels=[str(i) for i in np.arange(rows[0], rows[-1]+1)])
        if collims is not None:
            plt.xticks(np.arange(collims[-1]+1-collims[0]), labels=[str(i) for i in np.arange(cols[0], cols[-1]+1)])

        if show_aperture:
            aperture = self.aperture[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
            plt.imshow(np.ma.masked_where(aperture == False, aperture), origin='lower', cmap='binary', alpha=0.8)
        fig = plt.gcf()
        ax = plt.gca()
        plt.show()

        return fig, ax

    def plot_pixel(self, row=None, col=None, loc=None):
        """Plot the data (light curve) for a specified pixel.
        """
        flux = self.cutout_data.fluxes[:, row, col]
        plt.plot(self.cutout_data.time, flux, ".")

    def plot_pix_by_pix(self, data_type="raw", split=False, show_locations=True,
                        show_labels=True, fontsize=15, figsize=(12, 8), thin=1, marker=".", ms=1,
                        yaxis_nbins=6, ylabel_xloc = 0.065, zeroing=False):
        rows = np.arange(len(self.models))
        cols = np.arange(len(self.models[0]))
        fig, axs = plt.subplots(rows.size, cols.size, sharex=True, sharey=True, figsize=figsize, squeeze=False)
        for r in rows:
            for c in cols:
                ax = axs[rows[-1] - r, c]  # Needed to flip the rows so that they match origin='lower' setting
                if split:
                    yy = self.models[r][c].split_values_dict[data_type]
                    if (data_type == "cpm_subtracted_flux") & (zeroing == True):
                        yy = yy - self.models[r][c].split_values_dict["intercept_prediction"] 
                    for time, y in zip(self.split_times, yy):
                        ax.plot(time[::thin], y[::thin], marker, ms=ms)
                else:
                    y = self.models[r][c].values_dict[data_type]
                    if (data_type == "cpm_subtracted_flux") & (zeroing == True):
                        y = y - self.models[r][c].values_dict["intercept_prediction"] 
                    ax.plot(self.time[::thin], y[::thin], marker, ms=ms, color='k')
                if show_locations:
                    ax.text(x=0.98, y=0.98, s=f"[{self.models[r][c].row},{self.models[r][c].col}]", 
                            ha='right', va='top', transform=ax.transAxes)
                if show_labels:
                    if data_type == "raw":
                        y_label = r"Flux [$\mathrm{e^{-}s^{-1}}$]"
                    elif data_type == "cpm_subtracted_flux":
                        y_label = "De-trended Flux"
                    else:
                        y_label = "Normalized Flux"
                    fig.text(x=ylabel_xloc, y=0.5, s=y_label, fontsize=fontsize, rotation="vertical", va="center", rasterized=False)
                    fig.text(x=0.5, y=0.06, s="Time [BJD- 2457000]", fontsize=fontsize, ha="center", rasterized=False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=yaxis_nbins))
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return fig, axs

    def plot_pix_by_pix_via_mask(self, data_type="raw", split=False, show_locations=True,
                        show_labels=True, fontsize=15, figsize=(12, 8), thin=1, marker=".", ms=1,
                        yaxis_nbins=6, ylabel_xloc = 0.065, zeroing=False):
        plotted_rows = np.array([m[0].row for m in self.models])
        plotted_cols = np.array([m[0].col for m in self.models])
        min_r, max_r = np.min(plotted_rows), np.max(plotted_rows)
        min_c, max_c = np.min(plotted_cols), np.max(plotted_cols)
        fig_row_size = max_r - min_r + 1
        fig_col_size = max_c - min_c + 1
        plotted_rows = plotted_rows - min_r
        plotted_cols = plotted_cols - min_c
        fig, axs = plt.subplots(fig_row_size, fig_col_size, sharex=True, sharey=True, figsize=figsize, squeeze=False)
        row_max = np.max(plotted_rows)
        n_pixels = len(plotted_rows)
        for r, c, idx in zip(plotted_rows, plotted_cols, range(n_pixels)):
            ax = axs[row_max - r, c]
            if split:
                yy = self.models[idx][0].split_values_dict[data_type]
                if (data_type == "cpm_subtracted_flux") & (zeroing == True):
                    yy = yy - self.models[idx][0].split_values_dict["intercept_prediction"] 
                for time, y in zip(self.split_times, yy):
                    ax.plot(time[::thin], y[::thin], marker, ms=ms)
            else:
                y = self.models[idx][0].values_dict[data_type]
                if (data_type == "cpm_subtracted_flux") & (zeroing == True):
                    y = y - self.models[idx][0].values_dict["intercept_prediction"] 
                ax.plot(self.time[::thin], y[::thin], marker, ms=ms, color='k')
            if show_locations:
                ax.text(x=0.98, y=0.98, s=f"[{self.models[idx][0].row},{self.models[idx][0].col}]", 
                        ha='right', va='top', transform=ax.transAxes)
            if show_labels:
                if data_type == "raw":
                    y_label = r"Flux [$\mathrm{e^{-}s^{-1}}$]"
                elif data_type == "cpm_subtracted_flux":
                    y_label = "De-trended Flux"
                else:
                    y_label = "Normalized Flux"
                fig.text(x=ylabel_xloc, y=0.5, s=y_label, fontsize=fontsize, rotation="vertical", va="center", rasterized=False)
                fig.text(x=0.5, y=0.06, s="Time [BJD- 2457000]", fontsize=fontsize, ha="center", rasterized=False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=yaxis_nbins))
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return fig, axs


    def get_lc_matrix(self, data_type="cpm_subtracted_flux"):
        rows = np.arange(len(self.models))
        cols = np.arange(len(self.models[0]))
        lc_matrix = np.zeros((self.time.size, rows.size, cols.size))
        for r in rows:
            for c in cols:
                y = self.models[r][c].values_dict[data_type]
                lc_matrix[:, rows[-1] - r, c] = y
        return lc_matrix
    
    def make_animation(self, data_type="cpm_subtracted_flux", l=0, h=100, thin=5):
        lc_matrix = self.get_lc_matrix(data_type=data_type)
        fig, axes = plt.subplots(1, 1, figsize=(12, 12))
        ims = []
        for i in range(0, lc_matrix.shape[0], thin):  # pylint: disable=unsubscriptable-object
            im1 = axes.imshow(lc_matrix[i], animated=True,
                              vmin=np.percentile(lc_matrix, l), vmax=np.percentile(lc_matrix, h))  # origin="lower" is not used 
            ims.append([im1])
        fig.colorbar(im1, ax=axes, fraction=0.046)    
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
        return ani

    def rescale(self):
        for rowmod in self.models:
            for mod in rowmod:
                mod.rescale()

    def get_outliers(self, data_type="cpm_subtracted_flux", window=50, sigma=5, sigma_upper=None, sigma_lower=None):
        lc = self.get_aperture_lc(data_type=data_type, verbose=False)
        if sigma_upper is None:
            sigma_upper = sigma
        if sigma_lower is None:
            sigma_lower = sigma
        median_lc = median_filter(lc, size=window)
        median_subtracted_lc = lc - median_lc
        outliers = np.full(lc.shape, False)
        while True:
            std = np.std(median_subtracted_lc)
            clipped_upper = median_subtracted_lc > sigma_upper*std
            clipped_lower = median_subtracted_lc < -sigma_lower*std  # pylint: disable=invalid-unary-operand-type
            out = clipped_upper + clipped_lower
            if np.sum(out) == np.sum(outliers):
                break
            outliers += out
        return outliers

    def get_aperture_lc(self, data_type="raw", weighting=None, split=False, verbose=True):
        rows = np.arange(len(self.models))
        cols = np.arange(len(self.models[0]))
        if verbose:
            print(f"Summing over {rows.size} x {cols.size} pixel lightcurves. Weighting={weighting}")
        if split:
            aperture_lc = np.zeros_like(self.split_times, dtype=object)
        else:
            aperture_lc = np.zeros_like(self.time)
        medvals = np.zeros((len(rows), len(cols)))
        for r in rows:
            for c in cols:
                medvals[r][c] = self.models[r][c].cpm.target_median
        medvals /= np.nansum(medvals)
        for r in rows:
            for c in cols:
                if weighting == "median":
                    weight = medvals[r][c]
                elif weighting == None:
                    weight = 1.0
                if split:
                    aperture_lc += weight*self.models[r][c].split_values_dict[data_type]
                else:
                    aperture_lc += weight*self.models[r][c].values_dict[data_type]

        return aperture_lc

    def get_aperture_lc_via_mask(self, data_type="raw", weighting=None, split=False, verbose=True):
        plotted_rows = np.array([m[0].row for m in self.models])
        plotted_cols = np.array([m[0].col for m in self.models])
        min_r, max_r = np.min(plotted_rows), np.max(plotted_rows)
        min_c, max_c = np.min(plotted_cols), np.max(plotted_cols)
        fig_row_size = max_r - min_r + 1
        fig_col_size = max_c - min_c + 1
        rows_idx = plotted_rows - min_r
        cols_idx = plotted_cols - min_c
        n_pixels = len(plotted_rows)
        if verbose:
            print(f"Summing over {n_pixels} pixel lightcurves. Weighting={weighting}")
        if split:
            aperture_lc = np.zeros_like(self.split_times, dtype=object)
        else:
            aperture_lc = np.zeros_like(self.time)
        medvals = np.zeros(n_pixels)
        for idx in range(n_pixels):
            medvals[idx] = self.models[idx][0].cpm.target_median
        medvals /= np.nansum(medvals)
        for idx in range(n_pixels):
            if weighting == 'median':
                w = medvals[idx]
            elif weighting == None:
                w = 1.0
            if split:
                aperture_lc += w*self.models[idx][0].split_values_dict[data_type]
            else:
                aperture_lc += w*self.models[idx][0].values_dict[data_type]
        return aperture_lc
    
    def get_aperture_flux_errors(self):
        flux_errs = np.array(self.flux_errs).reshape(-1, self.time.size)
        ferr2 = flux_errs**2
        apt_ferr = np.sqrt(np.sum(ferr2, axis=0))
        return apt_ferr

    def _calc_cdpp(self, flux, **kwargs):
        return lk.TessLightCurve(flux=flux+1).estimate_cdpp(**kwargs)

    def calc_min_cpm_reg(self, cpm_regs, k, mask=None, **kwargs):
        cdpps = np.zeros((cpm_regs.size, k))
        for idx, reg in enumerate(cpm_regs):
            self.set_regs([reg])
            self.holdout_fit_predict(k, mask)
            apt_cpm_subtracted_lc = self.get_aperture_lc(split=True, data_type="cpm_subtracted_flux", verbose=False)
            split_cdpp = np.array([self._calc_cdpp(flux, **kwargs) for flux in apt_cpm_subtracted_lc])
            cdpps[idx] = split_cdpp
        section_avg_cdpps = np.average(cdpps, axis=1)
        section_sum_cdpps = np.sum(cdpps, axis=1)
        min_cpm_reg = cpm_regs[np.argmin(section_avg_cdpps)]
        fig, axs = plt.subplots(3, 1, figsize=(18, 15))
        for cpm_reg, cdpp in zip(cpm_regs, cdpps):
            axs[0].plot(np.arange(k)+1, cdpp, ".--", ms=10, label=f"Reg {cpm_reg}")
        axs[0].set_xlabel("k-th section of lightcurve", fontsize=15)
        axs[0].set_ylabel("CDPP", fontsize=20)
        # axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        for idx, cdpp in enumerate(cdpps.T):
            axs[1].plot(cpm_regs, cdpp, ".--", ms=10, label=f"Section {idx+1}")
        axs[1].set_xlabel("CPM Regularization Value [Precision]", fontsize=15)
        axs[1].set_ylabel("CDPP", fontsize=20)
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].legend()
        axs[2].plot(cpm_regs, section_avg_cdpps, ".-", c="k")
        axs[2].set_xlabel("CPM Regularization Value [Precision]", fontsize=15)
        axs[2].set_ylabel("Section Average CDPP", fontsize=20)
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")

        for ax in axs:
            ax.tick_params(labelsize="large")
        # axs[2].scatter(min_cpm_reg, section_avg_cdpps[np.where(cpm_regs == min_cpm_reg)], 
                    # marker="X", s=100, color="r", label="Minimum CPM Reg")
        # axs[2].legend()
        return (min_cpm_reg, cdpps)
    

    # def _lsq(self, y, m, reg_matrix, mask=None):
    #     if mask is not None:
    #         m = m[~mask]
    #         y = y[~mask]
    #     a = np.dot(m.T, m) + reg_matrix
    #     b = np.dot(m.T, y)
    #     w = np.linalg.solve(a, b)