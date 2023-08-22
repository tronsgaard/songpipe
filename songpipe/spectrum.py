from .frame import Frame, FrameList

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
    pass
