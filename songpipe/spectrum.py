"""Class definitions for extracted spectra"""
from glob import glob
from tqdm import tqdm
import numpy as np
import astropy.io.fits as fits

from pyreduce import echelle

from .frame import Frame, FrameList
from .misc import apply_limit

import logging
logger = logging.getLogger(__name__)

"""
This module contains the `Spectrum` class, which is currently identical with the `Frame` class.
"""

class Spectrum(Frame):
    
    def __init__(self, ech=None, header=None, filename=None):
        self._ech = ech  # PyReduce echelle object
        self._header = header
        self.filename = filename
        self.parent = None  # Not in use for extracted spectra

        # Load header from file
        if self._header is None:
            with fits.open(filename) as h:
                self._header = h[0].header  # Don't load data yet

        # Create empty header if necessary
        if self._header is None:
            self._header = fits.Header()

    def load_data(self):
        """Load BinTable data as PyReduce echelle object"""
        logger.debug(f'Loading PyReduce echelle object from "{self.filename}"')
        self._ech = echelle.read(self.filename, continuum_normalization=False, 
                                 barycentric_correction=False, radial_velociy_correction=False)

    def clear_data(self, keep_thead=False):
        """Clear cached data"""
        self._ech = None 
        if keep_thead is False:
            self._theader = None

    def save(self, filename=None):
        """Saves/updates ech file"""
        if filename is None:
            filename = self.filename
        if self._ech is None:
            self.load_data()
        logger.debug(f'Saving spectrum to {filename}')
        echelle.save(filename, self.header, **self._ech._data)

    def bin(self, binsize):
        """Create a binned spectrum"""
        ncol_new = self.ncol // binsize  # integer division
        # Make a binning function that reshapes the spectrum to extra dimension and then collapses the array
        binfunc = lambda x: np.sum(np.reshape(x[:,:(ncol_new * binsize)], (self.nord, ncol_new, binsize)), axis=2)
        
        # Initialise new spectrum
        spec = binfunc(self.spec)
        res = Spectrum(header=self.header, ech=echelle.Echelle(data=dict(spec=spec)))
        if self.sig is not None:
            res.sig = np.sqrt(binfunc(self.sig**2))
        if self.cont is not None:
            res.cont = binfunc(self.cont)
        if self.mask is not None: 
            res._set_ech_property('mask', binfunc(self.mask) > 0)

        # Update columm ranges
        if self.columns is not None:
            columns = self.columns // binsize
            columns[columns > ncol_new - 1] = ncol_new - 1
            res._set_ech_property('columns', columns)

        # Update wavelengths
        if self.wave is not None:
            res.wave = binfunc(self.wave) / binsize
        
        # TODO: Add something to header

        return res

    def _get_ech_property(self, name):
        if self._ech is None:
            self.load_data()
        return getattr(self._ech, name)
    
    def _set_ech_property(self, name, value):
        if self._ech is None:
            self.load_data()
        logger.debug(f'Setting ech property {name}')
        setattr(self._ech, name, value)

    @property
    def nord(self):
        return self._get_ech_property('nord')

    @property
    def ncol(self):
        return self._get_ech_property('ncol')

    @property
    def spec(self):
        return self._get_ech_property('spec')
    
    @spec.setter
    def spec(self, value):
        self._set_ech_property('spec', value)

    @property
    def sig(self):
        return self._get_ech_property('sig')

    @sig.setter
    def sig(self, value):
        self._set_ech_property('sig', value)

    @property
    def wave(self):
        return self._get_ech_property('wave')
    
    @wave.setter
    def wave(self, value):
        self._set_ech_property('wave', value)

    @property
    def cont(self):
        return self._get_ech_property('cont')
    
    @cont.setter
    def cont(self, value):
        self._set_ech_property('cont', value)
        
    @property
    def columns(self):
        return self._get_ech_property('columns')

    @property
    def mask(self):
        return self._get_ech_property('mask')


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

    def filter(self, *args, **kwargs):
        filtered_spectra = super().filter(*args, **kwargs)
        # Return new SpectrumList
        return SpectrumList(filtered_spectra)

    def append(self, spectrum):
        """Append a spectrum, if duplicate remove old entry"""
        if type(spectrum) != Spectrum:
            raise TypeError(f'Wrong type ({type(spectrum)}) - cannot append to SpectrumList')
        super().append(spectrum)
