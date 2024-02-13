import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut


def get_data(ra, dec, units="deg", size=64):
    c = SkyCoord(ra, dec, units=units)
    data_table = Tesscut.download_cutouts(c, size=size)
    return data_table


def plot_lightcurves(cpm):
    fig, axs = plt.subplots(2, 1, figsize=(18, 12))
    data = cpm.target_fluxes
    model = cpm.lsq_prediction
    res = data - cpm.cpm_prediction

    axs[0].plot(cpm.time, data, ".", color="k", label="Data", markersize=6)
    axs[0].plot(
        cpm.time, model, ".", color="C3", label="Model", markersize=4, alpha=0.6
    )
    if cpm.poly_params.shape != (0,):
        axs[0].plot(
            cpm.time,
            cpm.cpm_prediction,
            ".",
            color="C1",
            label="CPM",
            markersize=4,
            alpha=0.4,
        )
        axs[0].plot(
            cpm.time,
            cpm.poly_prediction,
            "-",
            color="C2",
            label="Poly",
            markersize=4,
            alpha=0.7,
        )

    axs[1].plot(cpm.time, res, ".-", label="Residual (Data - CPM)", markersize=3)

    for i in range(2):
        axs[i].legend(fontsize=15)
    plt.show()


def summary_plot(cpm, n=20, subtract_polynomials=False, save=False):
    """Shows a summary plot of a CPM fit to a pixel.

    The top row consists of three images: (left) image showing median values for each pixel,
    (middle) same image but the target pixel (the pixel we are trying to predict the lightcurve for),
    the exclusion region (area where predictor pixels CANNOT be chosen from), and the predictor pixels are shown,
    (right) same image but now the top-``n`` contributing pixels of the CPM are shown. The contribution is 
    defined as the absolute value of the coefficients of each predictor pixel. For example, the top 5 contributing pixels
    are the pixels which have the highest absolute coefficient values calculated when ``lsq`` is run.   

    """
    top_n_loc, top_n_mask = cpm.get_contributing_pixels(n)

    plt.figure(figsize=(18, 14))

    ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 3), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((4, 3), (0, 2), rowspan=2)

    ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=3)
    ax5 = plt.subplot2grid((4, 3), (3, 0), colspan=3)

    first_image = cpm.pixel_medians

    #     first_image = cpm.im_fluxes[0,:,:]
    ax1.imshow(
        first_image,
        origin="lower",
        vmin=np.nanpercentile(first_image, 10),
        vmax=np.nanpercentile(first_image, 90),
    )

    ax2.imshow(
        first_image,
        origin="lower",
        vmin=np.nanpercentile(first_image, 10),
        vmax=np.nanpercentile(first_image, 90),
    )
    ax2.imshow(cpm.excluded_pixels_mask, origin="lower", cmap="Set1", alpha=0.5)
    ax2.imshow(cpm.target_pixel_mask, origin="lower", cmap="binary", alpha=1.0)
    ax2.imshow(cpm.predictor_pixels_mask, origin="lower", cmap="binary_r", alpha=0.9)

    ax3.imshow(
        first_image,
        origin="lower",
        vmin=np.nanpercentile(first_image, 10),
        vmax=np.nanpercentile(first_image, 90),
    )
    ax3.imshow(cpm.excluded_pixels_mask, origin="lower", cmap="Set1", alpha=0.5)
    ax3.imshow(cpm.target_pixel_mask, origin="lower", cmap="binary", alpha=1.0)
    ax3.imshow(cpm.predictor_pixels_mask, origin="lower", cmap="binary_r", alpha=0.9)
    ax3.imshow(top_n_mask, origin="lower", cmap="Set1")

    data = cpm.rescaled_target_fluxes
    model = cpm.lsq_prediction
    if (subtract_polynomials == True) or (
        (subtract_polynomials == False) & (cpm.cpm_prediction is None)
    ):
        res = data - model
        plot_label = "Residual (Data - Model)"
    elif (subtract_polynomials == False) & (cpm.cpm_prediction is not None):
        res = data - (cpm.cpm_prediction + cpm.const_prediction)
        plot_label = "Residual (Data - CPM)"

    ax4.plot(cpm.time, data, ".", color="k", label="Data", markersize=4)
    ax4.plot(cpm.time, model, ".", color="C3", label="Model", markersize=4, alpha=0.4)

    if cpm.valid is not None:
        ax4.plot(
            cpm.time[~cpm.valid], data[~cpm.valid], "x", color="gray", label="Clipped"
        )

    if cpm.cpm_prediction is not None:
        ax4.plot(
            cpm.time,
            cpm.cpm_prediction,
            ".",
            color="C1",
            label="CPM",
            markersize=3,
            alpha=0.4,
        )
        ax4.plot(
            cpm.time,
            cpm.poly_prediction,
            ".",
            color="C2",
            label="Poly",
            markersize=3,
            alpha=0.4,
        )

    ax5.plot(cpm.time, res, ".-", label=plot_label, markersize=7)
    for dump in cpm.dump_times:
        ax5.axvline(dump, color="red", alpha=0.5)

    plt.suptitle(
        "N={} Predictor Pixels, Method: {}, L2Reg={}".format(
            cpm.num_predictor_pixels,
            cpm.method_predictor_pixels,
            "{:.0e}".format(cpm.cpm_regularization),
        ),
        y=0.89,
        fontsize=15,
    )

    ax1.set_title("Cmap set to 10%, 90% values of image")
    ax2.set_title("Target (White), Excluded (Red Shade), Predictors (Black)")
    ax3.set_title("Top N={} contributing pixels to prediction (Red)".format(n))

    ax4.set_title(
        "Lightcurves of Target Pixel {}".format((cpm.target_row, cpm.target_col))
    )
    # ax4.set_ylabel("Flux [e-/s]")
    ax4.set_ylabel("Relative Flux")
    ax4.legend(fontsize=12)

    ax5.set_xlabel("Time (BJD-2457000) [Day]", fontsize=12)
    # ax5.set_ylabel("Residual Flux [e-/s]")
    ax5.set_ylabel("Fractional Relative Fux")
    ax5.legend(fontsize=12)

    if save == True:
        plt.savefig(
            "cpm_target_{}_reg_{}_filename_{}.png".format(
                (cpm.target_row, cpm.target_col),
                "{:.0e}".format(cpm.cpm_regularization),
                cpm.file_name,
            ),
            dpi=200,
        )

def stitch_sectors(t1, t2, lc1, lc2, points=50):
    offset = np.ones((points, 1))
    m = np.block([
                 [t1[-points:].reshape(-1, 1), offset, np.zeros((points, 1))],
                 [t2[:points].reshape(-1, 1), np.zeros((points, 1)), offset]
        ])
    m[:,0] = m[:,0] - np.median(m[:,0])
    y = np.concatenate((lc1[-points:], lc2[:points]))
    a = np.dot(m.T, m)
    b = np.dot(m.T, y)
    params = np.linalg.solve(a, b)
    time = np.concatenate((t1, t2))
    diff = params[1] - params[2] + params[0]*(t2[0]-t1[-1])
    return (diff, params, time, np.concatenate((lc1, lc2+diff)))

# Maybe this function should be a method for the Source class.
# def get_outliers(lc, window=50, sigma=5, sigma_upper=None, sigma_lower=None):
    # if sigma_upper is None:
    #     sigma_upper = sigma
    # if sigma_lower is None:
    #     sigma_lower = sigma
    # median_lc = median_filter(lc, size=window)
    # median_subtracted_lc = lc - median_lc
    # outliers = np.full(lc.shape, False)
    # while True:
    #     std = np.std(median_subtracted_lc)
    #     clipped_upper = median_subtracted_lc > sigma_upper*std
    #     clipped_lower = median_subtracted_lc < -sigma_lower*std
    #     out = clipped_upper + clipped_lower
    #     if np.sum(out) == np.sum(outliers):
    #         break
    #     outliers += out
    # return outliers

# from IPython.display import HTML
# import matplotlib.animation as animation

# fig, axes = plt.subplots(1, 1, figsize=(12, 12))
# ims = []
# for i in range(0, lc_matrix.shape[0], 3):
#     im1 = axes.imshow(lc_matrix[i], animated=True,
#                         vmin=np.percentile(lc_matrix, 0), vmax=np.percentile(lc_matrix, 100))
#     ims.append([im1]);
# fig.colorbar(im1, ax=axes, fraction=0.046)
# # fig.colorbar(im2, ax=axes[1], fraction=0.046)
# # fig.colorbar(im3, ax=axes[2], fraction=0.046)
    
# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000);

# HTML(ani.to_jshtml())