"""Class definitions for extracted spectra"""
from glob import glob
from tqdm import tqdm
import astropy.io.fits as fits

from .frame import Frame, FrameList
from .misc import apply_limit

import logging
logger = logging.getLogger(__name__)

"""
This module contains the `Spectrum` class, which is currently identical with the `Frame` class.
"""

class Spectrum(Frame):
    
    def __init__(self, header=None, filename=None):
        self._header = header
        self._theader = None  # BinTable header (ext 1)
        self._tdata = None  # BinTable data
        self.filename = filename
        self.parent = None  # Not in use for extracted spectra
        self.file_handle = None

        # Load header from file
        if self._header is None:
            with fits.open(filename) as h:
                self._header = h[0].header  # Don't load data yet

        # Create empty header if necessary
        if self._header is None:
            self._header = fits.Header()

    def load_data(self, load_tdata=True):
        """Load BinTable data containing the spectrum etc."""
        logger.debug(f'Opening FITS BinTable from "{self.filename}"')
        with fits.open(self.filename) as h:
            logger.debug(f'Loading BinTable header')
            self._theader = h[1].header
            if load_tdata is True:
                logger.debug(f'Loading BinTable data')
                self._tdata = h[1].data

    def clear_data(self, keep_thead=False):
        """Clear cached data"""
        self._tdata = None 
        if keep_thead is False:
            self._theader = None

    @property
    def shape(self):
        """Get the dimensions (ncol x nord) from BinTable header"""
        if self._theader is None:
            self.load_data(load_tdata=False)
        i = 1
        while i < 100:
            try:
                ttype = self._theader[f'TTYPE{i}']
                if ttype == 'SPEC':
                    return self._theader[f'TDIM{i}']  # FIXME parse numbers
                else:
                    i += 1
            except KeyError:
                raise Exception('BinTable does not contain column SPEC')

    @property
    def ncol(self):
        return self.shape[0]  # Not working

    @property
    def nord(self):
        return self.shape[1]  # Not working

    @property
    def spec(self):
        pass

    @property
    def wave(self):
        pass


class SpectrumList(FrameList):
    
    def __init__(self, spectra):
        """Initialise SpectrumList object"""
        if isinstance(spectra, SpectrumList):
            self.spectra = spectra.spectra
        else:
            self.spectra = spectra

    @property
    def frames(self):
        """For compatibility with FrameList"""
        return self.spectra

    @classmethod
    def from_files(cls, files, limit=None, silent=False):
        files = apply_limit(files, limit)
        spectra = [Spectrum(filename=f) for f in tqdm(files, disable=silent)]
        return SpectrumList(spectra)

    @classmethod
    def from_filemask(cls, filemask, limit=None, silent=False):
        files = glob(filemask)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found: {filemask}')
        return cls.from_files(files, limit=limit, silent=silent)
