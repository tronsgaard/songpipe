"""Class definitions for extracted spectra"""
from glob import glob
from tqdm import tqdm
from .frame import Frame, FrameList
from .misc import apply_limit

"""
This module contains the `Spectrum` class, which is currently identical with the `Frame` class.
"""

class Spectrum(Frame):
    
    @property
    def nord(self):
        pass

    @property
    def npix(self):
        pass

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
