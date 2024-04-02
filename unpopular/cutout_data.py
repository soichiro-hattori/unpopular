import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Tesscut
from astropy.io import fits
from astropy.wcs import WCS
import lightkurve as lk


class CutoutData(object):
    """Object containing the data and additional attributes used in the TESS CPM model.

    Args:
        path (str): path to file
        remove_bad (bool): If ``True``, remove the data points that have been flagged by the TESS team. Default is ``True``.
        verbose (bool): If ``True``, print statements containing information. Default is ``True``.
        provenance (str): If ``TessCut``, the image being passed through is a TessCut cutout. If ``eleanor``, it is an eleanor postcard.

    """

    def __init__(self, path, remove_bad=True, verbose=True, 
                 provenance='TessCut', quality=None, bkg_subtract=False, bkg_n=100,
                 time_path=None, flux_path=None, ferr_path=None):
        try:
            self.file_path = path
            self.file_name = path.split("/")[-1]
        except AttributeError:
            print("Locally reading in files")
            self.file_path = None
            self.file_name = None
        
        if provenance == 'TessCut':
            s = self.file_name.split("-")
            self.sector = s[1].strip("s").lstrip("0")
            self.camera = s[2]
            self.ccd = s[3][0]

            with fits.open(path, mode="readonly") as hdu:
                self.time = hdu[1].data["TIME"]  # pylint: disable=no-member
                self.fluxes = hdu[1].data["FLUX"]  # pylint: disable=no-member
                self.flux_errors = hdu[1].data["FLUX_ERR"]  # pylint: disable=no-member
                if quality is None:
                    self.quality = hdu[1].data["QUALITY"]  # pylint: disable=no-member
                else:
                    self.quality = quality
                try:
                    self.wcs_info = WCS(hdu[2].header)  # pylint: disable=no-member
                except Exception as inst:
                    print(inst)
                    print("WCS Info could not be retrieved")
        
        elif provenance == 'eleanor':
            with fits.open(path, mode="readonly") as hdu:
                self.sector = int(hdu[2].header["SECTOR"])  # pylint: disable=no-member
                self.camera = int(hdu[2].header["CAMERA"])  # pylint: disable=no-member
                self.ccd    = int(hdu[2].header["CCD"])  # pylint: disable=no-member

                self.time = (hdu[1].data['TSTART'] + hdu[1].data['TSTOP'])/2  # pylint: disable=no-member
                self.fluxes = hdu[2].data  # pylint: disable=no-member
                self.flux_errors = hdu[3].data  # pylint: disable=no-member
                if quality is None:
                    self.quality = hdu[1].data['QUALITY']  # pylint: disable=no-member
                else:
                    self.quality = quality

                try:
                    self.wcs_info = WCS(hdu[2].header)  # pylint: disable=no-member
                except Exception as inst:
                    print(inst)
                    print("WCS Info could not be retrieved")

        elif provenance == 'local':
            self.sector = None
            self.camera = None
            self.ccd = None

            self.time = np.load(time_path)
            self.fluxes = np.load(flux_path)
            self.flux_errors = np.load(ferr_path)

            self.quality = np.zeros_like(self.time)
                    
        else:
            raise ValueError('Data provenance not understood. Pass through TessCut, eleanor, or local')

        self.flagged_times = self.time[self.quality > 0]
        # If remove_bad is set to True, we'll remove the values with a nonzero entry in the quality array
        if remove_bad == True:
            bool_good = self.quality == 0
            if verbose == True:
                print(
                    f"Removing {np.sum(~bool_good)} bad data points "
                    f"(out of {np.size(bool_good)}) using the TESS provided QUALITY array"
                )
            self.time = self.time[bool_good]
            self.fluxes = self.fluxes[bool_good]
            self.flux_errors = self.flux_errors[bool_good]

        # basic background correction based on subtracting median flux light curve of 500 faintest pixels
        if bkg_subtract:
            print("Performing initial basic background subtraction.")
            self.flux_medians = np.nanmedian(self.fluxes, axis=0)
            self.faint_pixel_locations = np.unravel_index(np.argpartition(self.flux_medians.ravel(),bkg_n)[:bkg_n], 
                                                     self.flux_medians.shape)
            self.faint_pixel_lcs = self.fluxes[:, self.faint_pixel_locations[0], self.faint_pixel_locations[1]]
            self.bkg_estimate = np.nanmedian(self.faint_pixel_lcs, axis=1)
            assert self.time.shape == self.bkg_estimate.shape
            self.fluxes -= self.bkg_estimate.reshape(-1, 1, 1)

        # We're going to precompute the pixel lightcurve medians since it's used to set the predictor pixels
        # but never has to be recomputed. np.nanmedian is used to handle images containing NaN values.
        self.flux_medians = np.nanmedian(self.fluxes, axis=0)
        self.cutout_sidelength_x = self.fluxes[0].shape[0]
        self.cutout_sidelength_y = self.fluxes[0].shape[1]
        
        self.flattened_flux_medians = self.flux_medians.reshape(
            self.cutout_sidelength_x * self.cutout_sidelength_y
        )
        # We rescale the fluxes by dividing by the median and then centering them around zero.
        self.normalized_fluxes = (self.fluxes / self.flux_medians) - 1
        self.flattened_normalized_fluxes = self.normalized_fluxes.reshape(
            self.time.shape[0], 
            self.cutout_sidelength_x * self.cutout_sidelength_y
        )

        self.normalized_flux_errors = self.flux_errors / self.flux_medians



