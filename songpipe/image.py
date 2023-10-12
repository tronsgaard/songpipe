from .frame import Frame, FrameList

from os.path import basename, dirname
from os import makedirs
from glob import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm

from .misc import apply_limit, header_insert

import logging
logger = logging.getLogger(__name__)

"""
METHODS FOR COMBINING IMAGES
"""
def median_combine(images, nallocate=10, silent=False):
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
            h = im.file_handle
            x[i] = h[im.ext].data[start:stop, :]
            t.update()  # Progress bar
        result[start:stop, :] = np.median(x, axis=0)

    t.close()  # Progress bar

    # Close all files
    for im in images:
        im.close_file()

    # TODO: Consider running a manual garbage collection here, to close any lingering memmap file handles

    header = fits.Header()
    header = header_insert(header, key='EXPTIME', value=np.median([im.exptime for im in images]))
    return Image(data=result, header=header)


def mean_combine(images, silent=False):
    """Mean combine a list of 2D images"""
    data = [im.data for im in tqdm(images, disable=silent)]
    return Image(data=np.mean(data, axis=0))

"""
CLASS DEFINITIONS
"""

class Image(Frame):
    """Represents a normal, single FITS image"""

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
        return (self.header['NAXIS1'], self.header['NAXIS2'])

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
    def load_data(self):
        """Load data from file"""
        logger.info(f'Loading FITS data from "{self.filename}" (ext {self.ext})')
        self.open_file()
        self._data = self.file_handle[self.ext].data
        self.close_file()

    def open_file(self, memmap=False):
        """Open a file handle """
        if self.filename is not None:
            self.file_handle = fits.open(self.filename, memmap=memmap)
            return self.file_handle
        else:
            raise ValueError('No such file!')

    def close_file(self):
        """Close open file handle"""
        try:
            self.file_handle.close()
            self.file_handle = None  # If we don't do this, the memory allocation seems to persist, 
                                     # even after closing the file and deleting other references to 
                                     # the data, leading to a memory leak when working on many files.
        except AttributeError:
            pass

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
    

class ImageList(FrameList):
    def __init__(self, images):
        self.images = images
        try:
            self.image_class = type(self.images[0])  # Assume at least one image in list
        except IndexError:
            self.image_class = Image

        # Check that all images are same type
        for im in self.images:
            assert type(im) == self.image_class

    @classmethod
    def from_files(cls, files, image_class=Image, limit=None, silent=False):
        files = apply_limit(files, limit)
        images = [image_class(filename=f) for f in tqdm(files, disable=silent)]
        return ImageList(images)

    @classmethod
    def from_filemask(cls, filemask, image_class=Image, limit=None, silent=False):
        return cls.from_files(glob(filemask), image_class=image_class, limit=limit, silent=silent)

    @property
    def files(self):
        return [im.filename for im in self.images]

    def __len__(self):
        """Provides len() compatibility"""
        return len(self.images)

    def __iter__(self):
        return self.images.__iter__()

    def __getitem__(self, item):
        try:
            return self.images.__getitem__(int(item))
        except ValueError:
            # If the int() cast fails
            pass
        # Get item by filename
        result = self.filter(filename_contains=item)
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            raise KeyError(f'Multiple files matching: "{item}"')
        else:
            raise KeyError(f'No such filename in list: {item}')

    def list(self, add_keys=None, outfile=None, silent=False):
        """
        Print a pretty list of filenames and some fits keywords.
        Add list of keys by using add_keys=['KEY1', 'KEY2', ...]
        Print to txt file by setting outfile=<path/to/file>
        """
        if len(self) == 0:
            return

        if isinstance(add_keys, str):
            add_keys = [add_keys]
        elif add_keys is None:
            add_keys = []

        buffer = []  # To be filled with all data before printing
        widths = {}  # Column widths
        for im in self.images:
            d = {}  # Dictionary of data for this image
            if im.filename is not None:
                d['filename'] = basename(im.filename)
            else:
                d['filename'] = ''
            d['image_type'] = im.type
            d['mode'] = im.mode
            d['exptime'] = im.exptime
            # Display additional keywords
            for k in add_keys:
                try:
                    d[k] = im.header[k]
                except KeyError:
                    d[k] = ''
            # Display object last
            d['object'] = im.object
            # Append row to buffer
            buffer.append(d)
            # Update dict of column widths
            for k in d.keys():
                widths[k] = max((len(str(d[k])), widths.get(k, 0),))
        
        # Define headers and update widths
        headers = {k: k.title() for k in widths.keys()}
        headers['image_type'] = 'Type'
        headers['exptime'] = 'Exp'
        widths = {k:max((len(headers[k]), w)) for k,w in widths.items()}
        
        # Define format string
        fmt = '  '.join([f'{{{k}:<{w}}}' for k, w in widths.items()]) + '\n'
        
        # Generate header line
        lines = []
        lines.append(fmt.format(**headers))
        
        # Generate table rows
        for d in buffer:
            lines.append(fmt.format(**d))

        # Export to file if requeted
        if outfile is not None:
            try:
                with open(outfile, 'w') as h:
                    h.writelines(lines)
            except IOError as e:
                logger.error('Could not export nightlog to txt file. Continuing..')
                logger.error(e)
        # Print
        if silent is not True:
            print('------------------------')
            for l in lines:
                print(l, end='')
            print('------------------------')
            print(f'Total: {len(self.images)}')
            print('------------------------')


    def summary(self):
        """Print summary of image types in the list"""
        count = {}
        for im in self.images:
            count[im.type] = count.get(im.type, 0) + 1

        for k, n in count.items():
            print(k, n)

    def get_exptimes(self, threshold=0.1):
        """
        Return a list of all exptimes, excluding bias images and times shorter than `threshold` (default: 0.1 s)
        """
        exptimes = np.unique([im.exptime for im in self.images if im.type != 'BIAS'])
        exptimes = exptimes[exptimes > threshold]
        return exptimes.tolist()

    def filter(self, object_contains=None, object_exact=None, filename_contains=None, filename_exact=None,
               image_type=None, image_type_exclude=None, mode=None, exptime=None, exptime_tol=0.1, 
               limit=None):
        """
        Filter list by various criteria and return result as a new list

        Arguments:
            object_contains :   Filter by partial object name
            object_exact :      Filter by exact object name
            filename_contains : Filter by partial filename
            filename_exact :    Filter by exact filename
            image_type :        Filter by image type or list of image types (e.g. `BIAS`, `STAR`)
            image_type_exclude: Exclude image type or list of image types
            mode :              Filter by instrument mode or list of modes (e.g. `F1`)
            exptime :           Filter by exposure time
            exptime_tol :       Tolerance used for exposure time filter (default: 0.1 s)
            limit :             Limit number of frames returned (default: unlimited)
        """

        def _ensure_list(x):
            """Ensure that x is a list/iterable"""
            if isinstance(x, str) or not hasattr(x, '__iter__'):
                x = [x]
            return x
        
        mask = [True] * len(self)
        for k, im in enumerate(self.images):
            if object_contains is not None and object_contains not in im.object:
                mask[k] = False
                continue
            if object_exact is not None and object_exact != im.object:
                mask[k] = False
                continue
            if filename_contains is not None and (im.filename is None or filename_contains not in im.filename):
                mask[k] = False
                continue
            if filename_exact is not None and (im.filename is None or filename_exact != im.filename):
                mask[k] = False
                continue
            if image_type is not None:
                image_type = _ensure_list(image_type)
                if im.type not in image_type:
                    mask[k] = False
                    continue
            if image_type_exclude is not None:
                image_type_exclude = _ensure_list(image_type_exclude)
                if im.type in image_type_exclude:
                    mask[k] = False
                    continue
            if mode is not None:
                mode = _ensure_list(mode)
                if im.mode not in mode:
                    mask[k] = False
                    continue
            if exptime is not None and np.abs(exptime - im.exptime) > exptime_tol:
                mask[k] = False
                continue
        # Apply mask
        images = np.array(self.images)[mask].tolist()
        # Apply limit
        images = apply_limit(images, limit)
        # Return new ImageList
        return ImageList(images)
    
    def count(self, **kwargs):
        """Passes all arguments to self.filter() and counts the number of frames returned"""
        res = self.filter(**kwargs)
        return len(res)

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
    