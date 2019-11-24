import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lc

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
        # print("Assuming you're interested in the central set of pixels")
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

    def holdout_fit_predict(self, k=10, mask=None):
        # if self.models is None:
        #     print("Please set the aperture first.")
        predictions = []
        fluxes = []
        detrended_lcs = []
        for row_models in self.models:
            row_predictions = []
            row_fluxes = []
            # row_detrended_lcs = []
            for model in row_models:
                times, flux, pred = model.holdout_fit_predict(k, mask)
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
        return (times, fluxes, predictions)

    def plot_cutout(self, rowrange=None, colrange=None, l=10, h=90, show_aperture=False):
        if rowrange is None:
            # rows = slice(0, self.target_data.cutout_sidelength)
            rows = [0, self.target_data.cutout_sidelength]
        else:
            # rows = slice(rowrange[0], rowrange[1])
            rows = rowrange

        if colrange is None:
            # cols = slice(0, self.target_data.cutout_sidelength)
            cols = [0, self.target_data.cutout_sidelength]
        else:
            # cols = slice(colrange[0], colrange[1])
            cols = colrange
        full_median_image = self.target_data.flux_medians
        median_image = self.target_data.flux_medians[rows[0]:rows[-1], cols[0]:cols[-1]]
        plt.imshow(
            median_image,
            origin="lower",
            vmin=np.percentile(full_median_image, l),
            vmax=np.percentile(full_median_image, h),
        )
        if rowrange is not None:
            plt.xticks(np.arange(rowrange[-1]-rowrange[0]), labels=[str(i) for i in np.arange(rows[0], rows[-1])])
        if colrange is not None:
            plt.yticks(np.arange(colrange[-1]-colrange[0]), labels=[str(i) for i in np.arange(cols[0], cols[-1])])

        if show_aperture:
            aperture = self.aperture[rows[0]:rows[-1], cols[0]:cols[-1]]
            plt.imshow(np.ma.masked_where(aperture == False, aperture), origin='lower', cmap='binary', alpha=0.8)
        plt.show()

    def plot_pixel(self, row=None, col=None, loc=None):
        """Plot the data (lightcurve) for a specified pixel.
        """
        flux = self.target_data.fluxes[:, row, col]
        plt.plot(self.target_data.time, flux, ".")

    def plot_pix_by_pix(self, split=False, data_type="raw", figsize=(12, 8), thin=1):
        # if self.split_predictions is None:
            # self.holdout_fit_predict()
        rows = np.arange(len(self.models))
        cols = np.arange(len(self.models[0]))
        fig, axs = plt.subplots(rows.size, cols.size, sharex=True, sharey=True, figsize=figsize, squeeze=False)
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
                    elif data_type == "rescaled_cpm_subtracted_lc":
                        yy = self.models[r][c].rescaled_cpm_subtracted_lc
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
                    elif data_type == "rescaled_cpm_subtracted_lc":
                        y = self.models[r][c].rescaled_cpm_subtracted_lc
                    # elif data_type == "detrended_lc":
                    #     y = np.concatenate(self.models[r][c].split_detrended_lcs)
                    ax.plot(self.time[::thin], y[::thin], '.', c='k')
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def rescale(self):
        for rowmod in self.models:
            for mod in rowmod:
                mod.rescale()

    def get_aperture_lc(self, split=False, data_type="raw", verbose=True):
        rows = np.arange(len(self.models))
        cols = np.arange(len(self.models[0]))
        if verbose:
            print(f"Summing over {rows.size} x {cols.size} pixel lightcurves")
        if split:
            aperture_lc = np.zeros_like(self.split_times)
        else:
            aperture_lc = np.zeros_like(self.time)
        for r in rows:
            for c in cols:
                if split:
                    if data_type == "raw":
                        aperture_lc += np.array(self.models[r][c].split_fluxes)
                    elif data_type == "prediction":
                        aperture_lc += self.models[r][c].split_prediction
                    elif data_type == "cpm_prediction":
                        aperture_lc += self.models[r][c].split_cpm_prediction
                    elif data_type == "poly_model_prediction":
                        aperture_lc += self.models[r][c].split_poly_model_prediction
                    # elif data_type == "detrended_lc":
                    #     yy = self.models[r][c].split_detrended_lcs
                    elif data_type == "cpm_subtracted_lc":
                        aperture_lc += self.models[r][c].split_cpm_subtracted_lc
                    elif data_type == "rescaled_cpm_subtracted_lc":
                        aperture_lc += self.models[r][c].split_rescaled_cpm_subtracted_lc
                else:
                    if data_type == "raw":
                        aperture_lc += self.models[r][c].y
                    elif data_type == "prediction":
                        aperture_lc += self.models[r][c].prediction
                    elif data_type == "cpm_prediction":
                        aperture_lc += self.models[r][c].cpm_prediction
                    elif data_type == "poly_model_prediction":
                        aperture_lc += self.models[r][c].poly_model_prediction
                    elif data_type == "cpm_subtracted_lc":
                        aperture_lc += self.models[r][c].cpm_subtracted_lc
                    elif data_type == "rescaled_cpm_subtracted_lc":
                        aperture_lc += self.models[r][c].rescaled_cpm_subtracted_lc
                    # elif data_type == "detrended_lc":
                    #     y = np.concatenate(self.models[r][c].split_detrended_lcs)
        return aperture_lc

    def _calc_cdpp(self, flux, **kwargs):
        return lc.TessLightCurve(flux=flux+1).estimate_cdpp(**kwargs)

    def calc_min_cpm_reg(self, cpm_regs, k, **kwargs):
        cdpps = np.zeros((cpm_regs.size, k))
        for idx, reg in enumerate(cpm_regs):
            self.set_regs([reg])
            self.holdout_fit_predict(k)
            apt_cpm_subtracted_lc = self.get_aperture_lc(split=True, data_type="cpm_subtracted_lc", verbose=False)
            split_cdpp = np.array([self._calc_cdpp(flux) for flux in apt_cpm_subtracted_lc])
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