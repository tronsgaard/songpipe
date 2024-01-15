from .frame import Frame, FrameList

from os.path import basename, dirname
from os import makedirs
from glob import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm

from .misc import apply_limit, header_insert, evaluate_range, transpose_range

import logging
logger = logging.getLogger(__name__)

"""
METHODS FOR COMBINING IMAGES
"""
def propagate_header_info(images, header=None):
    """Helper function for preserving image type etc. in combined images"""
    if header is None:
        header = fits.Header()

    image_type = list(set([im.type for im in images]))  # get unique values using set()
    if len(image_type) == 1:
        header = header_insert(header, key='IMAGETYP', value=image_type[0])

    objname = list(set([im.object for im in images]))
    if len(objname) == 1:
        header = header_insert(header, key='OBJECT', value=objname[0])

    observatory = list(set([im.observatory for im in images]))
    if len(objname) == 1:
        header = header_insert(header, key='OBSERVAT', value=observatory[0])

    mode = list(set([im.mode for im in images]))
    if len(mode) == 1:
        header = header_insert(header, key='PL_MODE', value=mode[0])

    # Propagate start time 
    try:
        # Sort according to MJD and get the first
        sorted_images = sorted(images, key=lambda x : x.mjd_start)
        im0 = sorted_images[0]
        
        header = header_insert(header, key='DATE-OBS', value=im0.date_start, comment='UTC Start of first exposure')
        header = header_insert(header, key='JD-DATE', value=im0.jd_start, comment='JD_UTC Start of first exposure')
        header = header_insert(header, key='MJD-DATE', value=im0.mjd_start, comment='MJD_UTC Start of first exposure')
    except Exception as e:
        logger.error('The following exception occurred when trying to \
                     get the start time of the first exposure in the stack')
        logger.exception(e)

    return header

def median_combine(images, nallocate=1, silent=False):
    """Median combine a list of 2D images"""

    # Open all files
    for im in images:
        im.open_file(memmap=True)

    # Configure stripes
    height, width = im.shape  # Image dimension
    n = len(images)
    stripeheight = height // n * nallocate  # Allocate memory corresponding to `nallocate` frames
    nstripes = int(np.ceil(height / stripeheight))
    logger.info(f'Median combine: Stripe height = {stripeheight} px ({nstripes} stripes)')
    result = np.zeros((height, width))

    # Loop over stripes
    t = tqdm(total=n * nstripes, disable=silent)
    for k in range(0, height, stripeheight):
        start = k
        stop = min(k + stripeheight, height)

        # Loop over images
        x = np.zeros((n, stop - start, width))
        for i, im in tqdm(enumerate(images), leave=False, unit_scale=True, disable=silent):
            #x[i] = h[im.ext].data[start:stop, :]
            x[i] = im.get_data(yrange=(start, stop), memmap=True, leave_open=True)
            t.update()  # Progress bar
        result[start:stop, :] = np.median(x, axis=0)

    t.close()  # Progress bar

    # Close all files
    for im in images:
        im.close_file()

    # TODO: Consider running a manual garbage collection here, to close any lingering memmap file handles

    # Build new header
    header = propagate_header_info(images)
    # Calculate new exptime
    exptimes = [im.exptime for im in images]
    header = header_insert(header, key='EXPTIME', value=np.median(exptimes), comment='Combined exposure time (median)')
    # More header info
    header = header_insert(header, key='PL_NCOMB', value=len(images), comment='Number of combined images')
    header = header_insert(header, key='PL_CBMTD', value='median', comment='Combine method')
    return Image(data=result, header=header)


def mean_combine(images, silent=False):
    """Mean combine a list of 2D images"""
    # Compute the mean
    data = [im.data for im in tqdm(images, disable=silent)]
    result = data=np.mean(data, axis=0)
    # Build new header
    header = propagate_header_info(images)
    exptimes = [im.exptime for im in images]
    header = header_insert(header, key='EXPTIME', value=np.mean(exptimes))
    return Image(data=result, header=header)

"""
CLASS DEFINITIONS
"""

class Image(Frame):
    """Represents a normal, single FITS image"""

    # Defines the area of the raw image to load
    XWINDOW = (None, None)
    YWINDOW = (None, None)

    def __init__(self, header=None, data=None, filename=None, ext=0, parent=None):
        self._header = header
        self._data = data
        self._shape = None
        self.filename = filename
        self.ext = ext
        self.parent = parent  # Enables us to go back to a HighLowImage and look in the primary header
        self.file_handle = None

        # Get shape
        if self._data is not None:
            self._shape = data.shape

        # Load header from file
        if self._header is None and self._data is None:
            with fits.open(filename) as h:
                self._header = h[ext].header  # Don't load data yet
                if self._shape is None:
                    self._shape = h[ext].shape

        # Create empty header if necessary
        if self._header is None:
            self._header = fits.Header()

    @classmethod
    def combine(cls, images, combine_function, **kwargs):
        """Combine a list of Images into one Image"""
        return combine_function(images, **kwargs)  # Returns an image
    
    @property
    def data(self):
        """Return array of data"""
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def shape(self):
        """Return the shape of the data array"""
        #return (self.header['NAXIS1'], self.header['NAXIS2'])
        ymax, xmax = self._shape  # Original shape
        xwin = evaluate_range(self.XWINDOW, xmax)
        ywin = evaluate_range(self.YWINDOW, ymax)
        win_shape = (ywin[1]-ywin[0], xwin[1]-xwin[0])
        return win_shape 

    @property
    def bias_subtracted(self):
        try:
            return self.header['PL_BISUB']
        except KeyError:
            return False

    @property
    def dark_subtracted(self):
        try:
            return self.header['PL_DASUB']
        except KeyError:
            return False

    @property
    def gain_applied(self):
        try:
            return self.header['PL_GNAPL']
        except KeyError:
            return False

    # Data handling
    def subtract_overscan(self, data):
        """Subtracts the prescan/overscan level"""
        return data  # Do nothing (override with subclass)

    def get_data(self, xrange=None, yrange=None, scale=True, 
                 remove_overscan=True, memmap=True, leave_open=False):
        """
        Get data from cache or file, 
        apply scaling, subtract overscan, return the resulting image.
        x and y range refers to the physical image region, ignoring overscan
        """
        xrange = xrange or (None, None)  # Replaces None with (None, None)
        yrange = yrange or (None, None)
        
        # If the (full) image is already loaded in, just get what we need from self._data
        if self._data is not None and scale is True and remove_overscan is True:
            return self._data[slice(*yrange), slice(*xrange)]  # Scaling and overscan already applied
        
        # Otherwise, fetch from file (use memmap=True to avoid loading the entire file)
        h = self.open_file(memmap=memmap)
        if remove_overscan is True:
            # Evaluate window definition; get rid of None values
            ymax, xmax = self._shape  # Original shape of data, before cropping
            xwin = evaluate_range(self.XWINDOW, xmax)
            ywin = evaluate_range(self.YWINDOW, ymax)

            # Transpose xrange and yrange to window
            ymax, xmax = self.shape  # Shape of window
            xrange = evaluate_range(xrange, xmax)
            yrange = evaluate_range(yrange, ymax)

            xtrans = transpose_range(xrange, xwin)
            ytrans = transpose_range(yrange, ywin)

            # Crop
            data = h[self.ext].data[slice(*ytrans), slice(*xtrans)]

            # Subtract overscan
            data = self.subtract_overscan(data)
        else:
            data = h[self.ext].data[slice(*yrange), slice(*xrange)]
        # Apply scaling
        if scale:
            bzero = self.header.get('BZERO', 0)
            bscale = self.header.get('BSCALE', 1)
            data = bzero + data * bscale
        # Close file
        if leave_open is False:
            self.close_file()
        return data
        

    def load_data(self):
        """Load data from file to memory"""
        logger.info(f'Loading FITS data from "{self.filename}" (ext {self.ext})')
        self._data = self.get_data(memmap=False, leave_open=False)  # Closes handle

    def open_file(self, memmap=False):
        """Open a file handle (HDUList)"""
        if self.filename is None:
            raise ValueError('Filename is empty!')
        # If already open, check if it uses the same memmap setting
        if self.file_handle is not None and self.file_handle._file.memmap is not memmap:
            self.close_file()  # Sets to None
        # Open file handle
        if self.file_handle is None:
            self.file_handle = fits.open(self.filename, memmap=memmap, 
                                         do_not_scale_image_data=True)
        # Also returns existing handle if file was already open
        return self.file_handle

    def close_file(self):
        """Close open file handle"""
        try:
            self.file_handle.close()
            self.file_handle = None  # If we don't do this, the memory allocation seems to persist, 
                                     # even after closing the file and deleting other references to 
                                     # the data, leading to a memory leak when working on many files.
        except AttributeError as e:
            logger.warning(f'Cannot close file handle to {self.filename}. May already be closed. Continuing..')

    def clear_data(self):
        """Clear cached data"""
        self._data = None

    def make_hdulist(self, dtype=None):
        """Produce a HDULIst for saving files"""
        return fits.HDUList([fits.PrimaryHDU(data=self.data.astype(dtype), header=self.header)])

    def save_fits(self, out_filename, overwrite=False, dtype=None):
        """Save image to a FITS file"""
        hdulist = self.make_hdulist(dtype=dtype)
        logger.info(f'Saving to {out_filename}...')
        makedirs(dirname(out_filename), exist_ok=True)  # Ensure that output folder exists
        hdulist.writeto(out_filename, overwrite=overwrite)
        self.filename = out_filename

    # Transformations
    def subtract_bias(self, bias, inplace=False):
        """Subtract bias image"""
        assert self.bias_subtracted is False
        header = header_insert(self._header, 'PL_BISUB', True, 'Bias subtracted')
        data = self.data - bias.data
        if inplace:
            self._data, self._header = data, header
            self.filename = None
        else:
            return Image(header=header, data=data)

    def subtract_dark(self, dark, inplace=False):
        """Subtract dark image"""
        assert self.bias_subtracted
        assert self.dark_subtracted is False
        data = self.data - dark.data
        header = header_insert(self._header, 'PL_DASUB', True, 'Dark subtracted')
        if inplace:
            self._data, self._header = data, header
            self.filename = None
        else:
            return Image(header=header, data=data)

    def orient(self, flip_updown=False, flip_leftright=False, rotation=0, inplace=False):
        """Orient the image by flipping, then rotating"""
        data = self.data
        if flip_updown:
            data = np.flipud(data)
        if flip_leftright:
            data = np.fliplr(data)
        if rotation != 0:
            data = np.rot90(data, k=rotation // 90)
        header = header_insert(self._header, 'PL_ORIEN', True, 'Oriented')
        if inplace:
            self._data, self._header = data, header
            self.filename = None
        else:
            return Image(header=header, data=data)

    def apply_gain(self, gain_factor, inplace=False):
        """Apply gain factor to convert from ADUs to electrons"""
        assert self.bias_subtracted
        # assert self.dark_subtracted
        assert self.gain_applied is False
        data = self.data * gain_factor
        header = header_insert(self._header, 'PL_GNAPL', True, 'Gain applied')
        if inplace:
            self._data, self._header = data, header
            self.filename = None
        else:
            return Image(header=header, data=data)

    # Plotting
    def show(self, ax=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None):
        assert self.data is not None
        if ax is None:
            plt.figure(figsize=(7, 5))
        else:
            plt.sca(ax)
        plt.imshow(self.data, vmin=vmin, vmax=vmax)
        plt.gca().invert_yaxis()
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))

    def hist(self, ax=None, bins=100, **kwargs):
        assert self.data is not None
        if ax is None:
            plt.figure(figsize=(7, 5))
        else:
            plt.sca(ax)
        plt.hist(self.data.flatten(), bins=bins, **kwargs)


class HighLowImage(Image):
    """Represents an image with separate high- and low-gain channels"""
    def __init__(self, high_gain_image=None, low_gain_image=None, filename=None):
        if high_gain_image is not None or low_gain_image is not None:
            assert isinstance(high_gain_image, Image)
            assert isinstance(low_gain_image, Image)
            high_gain_image.parent = self
            low_gain_image.parent = self
        self.high_gain_image = high_gain_image
        self.low_gain_image = low_gain_image
        self.filename = filename
        self.ext = None
        self.parent = None
        self.file_handle = None
        self._data = None
        self._header = None

        # Load from file
        if self.high_gain_image is None and self.low_gain_image is None:
            self.high_gain_image = Image(filename=filename, ext=0, parent=self)
            self.low_gain_image = Image(filename=filename, ext=1, parent=self)

    @classmethod
    def combine(cls, images, combine_function, **kwargs):
        """Combine a list of HighLowImages into one HighLowImage"""
        logger.info('Combining high gain images...')
        high_gain_image = combine_function([im.high_gain_image for im in images], **kwargs)
        logger.info('Combining low gain images...')
        low_gain_image = combine_function([im.low_gain_image for im in images], **kwargs)
        return HighLowImage(high_gain_image, low_gain_image)  # Return a HighLowImage

    @staticmethod
    def _dual_plot(func_high, func_low, **kwargs):
        fig, ax = plt.subplots(ncols=2)
        fig.set_size_inches(12, 5)
        plt.sca(ax[0])
        func_high(ax=ax[0], **kwargs)
        plt.title('High gain')
        plt.sca(ax[1])
        func_low(ax=ax[1], **kwargs)
        plt.title('Low gain')

    def show(self, **kwargs):
        self._dual_plot(self.high_gain_image.show, self.low_gain_image.show, **kwargs)

    def hist(self, **kwargs):
        self._dual_plot(self.high_gain_image.hist, self.low_gain_image.hist, **kwargs)

    # Data handling
    def load_data(self):
        self.high_gain_image.load_data()
        self.low_gain_image.load_data()

    def clear_data(self):
        self.high_gain_image.clear_data()
        self.low_gain_image.clear_data()

    def make_hdulist(self, dtype=None):
        return fits.HDUList([
            fits.PrimaryHDU(data=self.high_gain_image.data.astype(dtype), header=self.high_gain_image.header),
            fits.ImageHDU(data=self.low_gain_image.data.astype(dtype), header=self.low_gain_image.header),
        ])

    @property
    def header(self):
        return self.high_gain_image.header

    @property
    def data(self):
        raise NotImplementedError

    # Transformations
    def subtract_bias(self, bias, inplace=False):
        assert self.bias_subtracted is False
        high_gain_image = self.high_gain_image.subtract_bias(bias.high_gain_image, inplace=inplace)
        low_gain_image = self.low_gain_image.subtract_bias(bias.low_gain_image, inplace=inplace)
        if inplace is False:
            return HighLowImage(high_gain_image=high_gain_image, low_gain_image=low_gain_image)

    def subtract_dark(self, dark, inplace=False):
        assert self.bias_subtracted
        assert self.dark_subtracted is False
        high_gain_image = self.high_gain_image.subtract_dark(dark.high_gain_image, inplace=inplace)
        low_gain_image = self.low_gain_image.subtract_dark(dark.low_gain_image, inplace=inplace)
        if inplace is False:
            return HighLowImage(high_gain_image=high_gain_image, low_gain_image=low_gain_image)

    def apply_gain(self, gain_high=0.78, gain_low=15.64, inplace=False):
        # electrons/ADU for HIGHGAIN and LOWGAIN image, respectively: [0.78, 15.64]
        assert self.bias_subtracted
        # assert self.dark_subtracted
        assert self.gain_applied is False
        high_gain_image = self.high_gain_image.apply_gain(gain_high, inplace=inplace)
        low_gain_image = self.low_gain_image.apply_gain(gain_low, inplace=inplace)
        if inplace is False:
            return HighLowImage(high_gain_image=high_gain_image, low_gain_image=low_gain_image)

    def orient(self, flip_updown=False, flip_leftright=False, rotation=0, inplace=False):
        high_gain_image = self.high_gain_image.orient(rotation=rotation, flip_updown=flip_updown, flip_leftright=flip_leftright)
        low_gain_image = self.low_gain_image.orient(rotation=rotation, flip_updown=flip_updown, flip_leftright=flip_leftright)
        if inplace is False:
            return HighLowImage(high_gain_image=high_gain_image, low_gain_image=low_gain_image)

    def merge_high_low(self, threshold=3000):
        assert self.bias_subtracted
        # assert self.dark_subtracted
        assert self.gain_applied

        high_gain_data = self.high_gain_image.data
        low_gain_data = self.low_gain_image.data

        mask = high_gain_data >= threshold  # TODO: What about saturated pixels?
        merged = high_gain_data.copy()
        merged[mask] = low_gain_data[mask]

        return Image(data=merged, header=self.header)

class QHYImage(Image):
    """
    Represents a CMOS image from the QHY detectors installed in 2024
    Mirrors the FITS structure with image and image header in a separate HDU.
    """
    XWINDOW = (24,9600)
    YWINDOW = (0,6388)


class AndorImage(Image):
    """Represents a CCD image with prescan/overscan regions"""
    YWINDOW = (20, 2068)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overscan_level = None

    def subtract_overscan(self, data):
        """Subtracts the prescan/overscan level from input data"""
        # TODO: Fit and subtract ramp or surface instead of constant offset
        if self._overscan_level is None:
            prescan = self.get_data(yrange=(5, 20), remove_overscan=False)
            overscan = self.get_data(yrange=(2068, 2083), remove_overscan=False)
            self._overscan_level = np.median(np.hstack(prescan, overscan))
        
        return data - self._overscan_level

class ImageList(FrameList):
    def __init__(self, images, image_class=None):
        """If argument `images` is an empty list, `image_class` must be specified"""
        if isinstance(images, ImageList):
            self.images = images.images
        else:
            self.images = list(images)

        if image_class is not None:
            self.image_class = image_class
        else:
            try:
                self.image_class = type(self.images[0])  # Assume at least one image in list
            except IndexError:
                raise Exception('Could not determine image class. \
                                Please supply at least one Image or use the `image_class` keyword argument')

        # Check that all images are same type
        for im in self.images:
            assert type(im) == self.image_class

    @classmethod
    def from_files(cls, files, image_class=Image, limit=None, silent=False):
        files = apply_limit(files, limit)
        images = [image_class(filename=f) for f in tqdm(files, disable=silent)]
        return ImageList(images)

    @classmethod
    def from_filemask(cls, filemask, ignore_list=None, image_class=Image, limit=None, silent=False):
        """Initialize ImageList from a glob-compatible filemask"""
        files = glob(filemask)
        nfiles = len(files)
        # Check files against ignore list 
        if ignore_list is not None:
            files_new = []
            for f in files:
                if basename(f) in ignore_list:
                    logger.info(f'File ignored: {f}')
                else:
                    files_new.append(f)
            files = files_new

        if len(files) == 0:
            raise FileNotFoundError(f'No images found: {filemask}')
        return cls.from_files(files, image_class=image_class, limit=limit, silent=silent)

    @property
    def frames(self):
        return self.images  # hack that ensures methods from FrameList will work
                            # Ideally self.images should be renamed
        
    def append(self, image):
        if type(image) != self.image_class:
            raise TypeError(f'Wrong image class ({self.image_class}) - cannot append to this ImageList')
        super().append(image)

    def append_from_files(self, files, limit=None, silent=False):
        """Add images from a list of filenames"""
        files = apply_limit(files, limit)
        for f in tqdm(files, disable=silent):
            image = self.image_class(filename=f)
            self.append(image)

    def append_from_filemask(self, filemask, limit=None, silent=False):
        """Add images from a filemask"""
        self.append_from_files(glob(filemask), limit=limit, silent=silent)

    def clear_data(self):
        """Clear all data from memory"""
        for im in self.images:
            im.clear_data()

    def filter(self, *args, **kwargs):
        filtered_images = super().filter(*args, **kwargs)
        # Return new ImageList
        return ImageList(filtered_images, image_class=self.image_class)

    def combine(self, method='median', **kwargs):
        """Combine all images in the list using specified method"""
        # Parse method name and get function
        if method == 'median':
            combine_function = median_combine
        elif method == 'mean':
            combine_function = mean_combine
        else:
            raise ValueError(f'Unknown method "{method}"!')
        # Call the function
        logger.info(f'Combining {len(self)} images using method "{method}".')
        result = self.image_class.combine(self.images, combine_function, **kwargs)
        logger.info('Combine done!')
        return result
    
