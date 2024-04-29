"""Custom instrument classes for pyreduce"""
import os.path
import numpy as np
import astropy.io.fits as fits
from pyreduce.instruments.common import Instrument

class SONGInstrument(Instrument):
    """
    Custom Instrument class for PyReduce
    """

    def __init__(self, name, modes=None):
        #:str: Name of the instrument (lowercase)
        #self.name = self.__class__.__name__.lower()
        self.name =  name
        #:dict: Information about the instrument
        self.info = {'modes': modes}

    def __str__(self):
        return self.name

    def get(self, key, header, mode, alt=None):
        #get = getter(header, self.info, mode)
        #return get(key, alt=alt)
        raise NotImplementedError

    def get_extension(self, header, mode):
        return 0

    def load_info(self):
        """
        Load static instrument information
        """
        return self.info

    def load_fits(self, fname, mode, extension=None, mask=None, header_only=False, dtype=None):
        """
        load fits file, REDUCE style

        primary and extension header are combined
        modeinfo is applied to header
        data is clipnflipped
        mask is applied

        Parameters
        ----------
        fname : str
            filename
        instrument : str
            name of the instrument
        mode : str
            instrument mode
        extension : int
            data extension of the FITS file to load
        mask : array, optional
            mask to add to the data
        header_only : bool, optional
            only load the header, not the data
        dtype : str, optional
            numpy datatype to convert the read data to

        Returns
        --------
        data : masked_array
            FITS data, clipped and flipped, and with mask
        header : fits.header
            FITS header (Primary and Extension + Modeinfo)

        ONLY the header is returned if header_only is True
        """

        mode = mode.upper()

        with fits.open(fname) as hdu:
            h_prime = hdu[0].header
            if extension is None:
                extension = self.get_extension(h_prime, mode)

            header = hdu[extension].header
            if extension != 0:
                header.extend(h_prime, strip=False)
            #header = self.add_header_info(header, mode)
            header["e_input"] = (os.path.basename(fname), "PyReduce input filename")
            header["e_xlo"] = (0, "PyReduce")
            header["e_xhi"] = (header["naxis1"], "PyReduce")
            header["e_ylo"] = (0, "PyReduce")
            header["e_yhi"] = (header["naxis2"], "PyReduce")
            header["e_gain"] = (self.info.get("gain", 1), "PyReduce gain (e/ADU)")  # e-/ADU
            header["e_readn"] = (self.info.get("readn", 0), "PyReduce read noise (e)")  # electrons
            header["e_drk"] = (self.info.get("dark", 0), "PyReduce dark current (e)")  # Dark current (electrons), used for uncertainty calculation

            if header_only:
                return header

            data = hdu[extension].data
            #data = clipnflip(data, header)

            if dtype is not None:
                data = data.astype(dtype)

            data = np.ma.masked_array(data, mask=mask)

            return data, header

    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header

        Parameters
        ----------
        header : fits.header, dict
            header to read/write info from/to
        mode : str
            instrument mode

        Returns
        -------
        header : fits.header, dict
            header with added information
        """

        #info = self.load_info()
        #get = getter(header, info, mode)

        #header["e_instru"] = get("instrument", self.__class__.__name__)
        #header["e_telesc"] = get("telescope", "")
        #header["e_exptim"] = get("exposure_time", 0)

        #jd = get("date")
        #if jd is not None:
        #    jd = Time(jd, format=self.info.get("date_format", "fits"))
        #    jd = jd.to_value("mjd")

        #header["e_orient"] = get("orientation", 0)
        # As per IDL rotate if orient is 4 or larger and transpose is undefined
        # the image is transposed
        #header["e_transp"] = get("transpose", (header["e_orient"] % 8 >= 4))

        #naxis_x = get("naxis_x", 0)
        #naxis_y = get("naxis_y", 0)

        #prescan_x = get("prescan_x", 0)
        #overscan_x = get("overscan_x", 0)
        #prescan_y = get("prescan_y", 0)
        #overscan_y = get("overscan_y", 0)

        #header["e_xlo"] = prescan_x
        #header["e_xhi"] = naxis_x - overscan_x

        #header["e_ylo"] = prescan_y
        #header["e_yhi"] = naxis_y - overscan_y

        #header["e_gain"] = get("gain", 1)
        #header["e_readn"] = get("readnoise", 0)

        #header["e_sky"] = get("sky", 0)
        #header["e_drk"] = get("dark", 0)
        #header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])

        #header["e_imtype"] = get("image_type")
        #header["e_ctg"] = get("category")

        #header["e_ra"] = get("ra", 0)
        #header["e_dec"] = get("dec", 0)
        #header["e_jd"] = jd

        #header["e_obslon"] = get("longitude")
        #header["e_obslat"] = get("latitude")
        #header["e_obsalt"] = get("altitude")

        #if info.get("wavecal_element", None) is not None:
        #    header["HIERARCH e_wavecal_element"] = get(
        #        "wavecal_element", info.get("wavecal_element", None)
        #    )
        return header

    def find_files(self, input_dir):
        raise NotImplementedError

    def get_expected_values(self, target, night, *args, **kwargs):
        raise NotImplementedError

    def populate_filters(self, files):
        raise NotImplementedError

    def apply_filters(self, files, expected, allow_calibration_only=False):
        raise NotImplementedError

    def sort_files(self, input_dir, target, night, *args, allow_calibration_only=False, **kwargs):
        raise NotImplementedError

    def get_wavecal_filename(self, header, mode, **kwargs):
        raise NotImplementedError

    def get_supported_modes(self):
        info = self.load_info()
        return info["modes"]

    def get_mask_filename(self, mode, **kwargs):
        return None

    def get_wavelength_range(self, header, mode, **kwargs):
        return self.get("wavelength_range", header, mode)

