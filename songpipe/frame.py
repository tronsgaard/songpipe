from os.path import basename
import astropy.io.fits as fits

from . import config
from .misc import construct_filename

import logging
logger = logging.getLogger(__name__)

"""
This module contains the Frame and FrameList classes, on which the Image/ImageList and Spectrum/SpectrumList classes are based.
"""

class Frame:
    """
    Base class for Image and Spectrum. 
    Parent-child structure allows for HighLowImage containing two Image objects
    """

    def __init__(self, header=None, data=None, filename=None, ext=0, parent=None):
        self._header = header
        self._data = data
        self.filename = filename
        self.ext = ext
        self.parent = parent  # Enables us to go back to a HighLowImage and look in the primary header
        self.file_handle = None

        # Load header from file
        if self._header is None and self._data is None:
            with fits.open(filename) as h:
                self._header = h[ext].header  # Don't load data yet

        # Create empty header if necessary
        if self._header is None:
            self._header = fits.Header()

    def get_header_value(self, key):
        """If header does not contain the key, go back and check the parent frame (e.g. a HighLowImage)"""
        try:
            return self.header[key]
        except KeyError:
            if self.parent is not None:
                return self.parent.get_header_value(key)
            else:
                raise

    @property
    def header(self):
        return self._header

    @property
    def object(self):
        return self.get_header_value('OBJECT')

    @property
    def exptime(self):
        return self.get_header_value('EXPTIME')

    @property
    def date_start(self):
        return self.get_header_value('DATE-OBS')
    
    @property
    def jd_start(self):
        return self.get_header_value('JD-DATE')
    
    @property
    def mjd_start(self):
        return self.get_header_value('MJD-DATE')

    @property
    def mjd_mid(self):
        """Get MJD_MID from header or calculate from exptime"""
        try:
            return self.get_header_value('MJD-MID')
        except KeyError:
            exptime_days = self.exptime/86400.
            return self.mjd_start + 0.5*exptime_days

    @property
    def type(self):
        return self.get_header_value('IMAGETYP')
    
    @property
    def observatory(self):
        """Get observatory name from FITS header"""
        try:
            return self.get_header_value('OBSERVAT')
        except KeyError:
            logger.warning(f'Missing keyword OBSERVAT: {self.filename}')
            return 'Unknown'

    @property
    def is_australia(self):
        """Check if this frame comes from Australia"""
        return True if self.observatory == 'Mt. Kent' else False
    
    @property
    def is_tenerife(self):
        """Check if this frame comes from Tenerife"""
        return True if self.observatory == 'Observatorio del Teide' else False

    @property
    def mode(self):
        """
        Determine the instrument mode: 
           Mt. Kent:    F1, F2, F12, SLIT, DARK, UNKNOWN
           Tenerife:    SLITx, DARK, UNKNOWN
        """
        try:
            # First look for SONGPIPE header keyword
            return self.get_header_value('PL_MODE')
        except KeyError:
            # Otherwise, derive from rest of header
            if self.type in ('DARK', 'BIAS'):
                return 'DARK'
            # Tenerife:
            if self.is_tenerife:
                slit = self.get_header_value('SLIT')
                return f'SLIT{slit}'
            # Australia:
            if self.is_australia:
                if self.type == 'FLAT' and self.get_header_value('LIGHTP') == 1:
                    return 'SLIT'
                if self.type in ('FLAT', 'FLATI2', 'THAR', 'FP'):
                    # Check telescope shutters
                    tel1 = self.get_header_value('TEL1_S')
                    tel2 = self.get_header_value('TEL2_S')
                    if tel1 == 1 and tel2 == 1:
                        return 'F12'
                    if tel1 == 1:
                        return 'F1'
                    if tel2 == 1:
                        return 'F2'
            # In any other case (all observations):
            # Note: This includes all star spectra from Mt. Kent, as the header 
            #       doesn't tell if there is starlight in both fibres.
            return 'UNKNOWN'

    # Misc
    def construct_filename(self, **kwargs):
        """
        Wrapper for the `construct_filename()` method.
        Passes object name from FITS header, if config.OBJECT_IN_FILENAME is True
        """
        if 'object' in kwargs:
            obj = kwargs.pop('object')  # Removes item from kwargs
        elif config.OBJECT_IN_FILENAME is True:
            obj = self.object
        else:
            obj = None
        return construct_filename(basename(self.get_header_value('FILE')), object=obj, **kwargs)

class FrameList:
    """ImageList and SpectrumList inherit shared properties and methods from this class"""
    pass 