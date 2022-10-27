from os.path import basename, splitext, dirname, exists
from os import makedirs
from glob import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
HELPER FUNCTIONS
"""


def construct_filename(orig_filename, object=None, prepared=False, extracted=False, fiber=None,
                       prefix=None, suffix=None):
    """Construct a standardized filename based on the properties of the file"""
    filename, ext = splitext(orig_filename)
    if object is not None:
        filename += '_' + object
    # Prepend stuff
    if fiber is not None:
        filename = f'F{fiber:.0f}'
    if prefix is not None:
        filename = prefix + '_' + filename
    # Append stuff
    if prepared:
        filename += '_prep'
    elif extracted:
        filename += '_extr'
    if suffix is not None:
        filename += '_' + suffix
    if ext == '':
        ext = '.fits'
    return filename + ext


def header_insert(hdr, key, value=None, comment=''):
    """Keep the header organized by grouping all pipeline keywords in a section"""
    hdr = hdr.__copy__()
    SECTION_HEADER = ('---PL---', '----PIPELINE----', '-------------------------------------')
    get_keys = lambda: list(hdr.keys())  # Get updated list of keys from header
    try:
        start = get_keys().index(SECTION_HEADER[0])
    except ValueError:
        hdr.set(*SECTION_HEADER)
        start = get_keys().index(SECTION_HEADER[0])
    end = start

    # Determine end of section
    in_section = True
    keys = get_keys()
    while in_section is True:
        if end + 1 >= len(keys) or keys[end + 1][0] == '-' or keys[end + 1] == 'COMMENT':
            in_section = False
        else:
            end += 1

    # Insert header key/value
    hdr.insert(end, (key, value, comment), after=True)
    return hdr


def apply_limit(array, limit):
    """SQL-like limit syntax"""
    if not hasattr(limit, '__iter__'):
        limit = (limit,)
    return array[slice(*limit)]


"""
METHODS FOR COMBINING IMAGES
"""


def median_combine(images, nallocate=10, verbose=True):
    """Median combine a list of 2D images"""

    # Open all files
    for im in images:
        im.open_file(memmap=True)

    # Configure stripes
    height, width = im.shape  # Image dimension
    n = len(images)
    stripeheight = height // n * nallocate  # Allocate memory corresponding to `nallocate` frames
    nstripes = int(np.ceil(height / stripeheight))
    print(f'Median combine: Stripe height = {stripeheight} px ({nstripes} stripes)')
    result = np.zeros((height, width))

    # Loop over stripes
    t = tqdm(total=n * nstripes)
    for k in range(0, height, stripeheight):
        start = k
        stop = min(k + stripeheight, height)

        # Loop over images
        x = np.zeros((n, stop - start, width))
        for i, im in tqdm(enumerate(images), leave=False, unit_scale=True):
            h = im.file_handle
            x[i] = h[0].data[start:stop, :]
            t.update()  # Progress bar
        result[start:stop, :] = np.median(x, axis=0)

    t.close()  # Progress bar

    # Close all files
    for im in images:
        im.close_file()

    header = fits.Header()
    header = header_insert(header, key='EXPTIME', value=np.median([im.exptime for im in images]))
    return Image(data=result, header=header)


def mean_combine(images):
    """Mean combine a list of 2D images"""
    data = [im.data for im in tqdm(images)]
    return Image(data=np.mean(data, axis=0))


"""
CLASS DEFINITIONS
"""


class Image:
    """Represents a normal, single FITS image"""

    def __init__(self, header=None, data=None, filename=None, ext=0):
        super().__init__()
        self._header = header
        self._data = data
        self.filename = filename
        self.ext = ext
        self.file_handle = None

        # Load header from file
        if self._header is None and self._data is None:
            with fits.open(filename) as h:
                self._header = h[ext].header  # Don't load data yet

        # Create empty header if necessary
        if self._header is None:
            self._header = fits.Header()

    @classmethod
    def combine(cls, images, combine_function, **kwargs):
        """Combine a list of Images into one Image"""
        return combine_function(images, **kwargs)  # Returns an image

    @property
    def header(self):
        return self._header

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
    def object(self):
        return self.header['OBJECT']

    @property
    def exptime(self):
        return self.header['EXPTIME']

    @property
    def type(self):
        return self.header['IMAGETYP']

    @property
    def mode(self):
        """Returns the instrument mode, currently (MtKent): F1, F2, or F12, SLIT, DARK, UNKNOWN"""
        if self.type in ('DARK', 'BIAS'):
            return 'DARK'
        if self.type == 'FLAT' and self.header['LIGHTP'] == 1:
            return 'SLIT'
        if self.type in ('FLAT', 'FLATI2', 'THAR', 'FP'):
            # Check telescope shutters
            tel1 = self.header['TEL1_S']
            tel2 = self.header['TEL2_S']
            if tel1 == 1 and tel2 == 1:
                return 'F12'
            if tel1 == 1:
                return 'F1'
            if tel2 == 1:
                return 'F2'
        # In any other case (including all observations):
        return 'UNKNOWN'  # FIXME: Figure out a way to detect if there is light in both fibres

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
        print(f'Loading FITS data from "{self.filename}" (ext {self.ext})')
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
        print(f'Saving to {out_filename}...')
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

    # Misc
    def construct_filename(self, **kwargs):
        return construct_filename(basename(self.header['FILE']), object=self.object, **kwargs)


class HighLowImage(Image):
    def __init__(self, high_gain_image=None, low_gain_image=None, filename=None):
        if high_gain_image is not None or low_gain_image is not None:
            assert isinstance(high_gain_image, Image)
            assert isinstance(low_gain_image, Image)
        self.high_gain_image = high_gain_image
        self.low_gain_image = low_gain_image
        self.filename = filename
        self.ext = None
        self.file_handle = None
        self._data = None
        self._header = None

        # Load from file
        if self.high_gain_image is None and self.low_gain_image is None:
            self.high_gain_image = Image(filename=filename, ext=0)
            self.low_gain_image = Image(filename=filename, ext=1)

    @classmethod
    def combine(cls, images, combine_function):
        """Combine a list of HighLowImages into one HighLowImage"""
        print('Combining high gain images...')
        high_gain_image = combine_function([im.high_gain_image for im in images])
        print('Combining low gain images...')
        low_gain_image = combine_function([im.high_gain_image for im in images])
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
            fits.ImageHDU(data=self.high_gain_image.data.astype(dtype), header=self.high_gain_image.header),
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


class ImageList():
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
    def from_files(cls, files, image_class=Image, limit=None):
        files = apply_limit(files, limit)
        images = [image_class(filename=f) for f in tqdm(files)]
        return ImageList(images)

    @classmethod
    def from_filemask(cls, filemask, image_class=Image, limit=None):
        return cls.from_files(glob(filemask), image_class=image_class, limit=limit)

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

    def list(self, add_keys=None):
        """
        Print a pretty list of filenames and some fits keywords.
        Add list of keys by using add_keys=['KEY1', 'KEY2', ...]
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
        fmt = '  '.join([f'{{{k}:<{w}}}' for k, w in widths.items()])
        # Print headers
        print(fmt.format(**headers))
        # Print table rows
        for d in buffer:
            print(fmt.format(**d))

    def summary(self):
        """Print summary of image types in the list"""
        count = {}
        for im in self.images:
            count[im.type] = count.get(im.type, 0) + 1

        for k, n in count.items():
            print(k, n)

    def get_exptimes(self, threshold=0.1):
        """Return a list of exptimes"""
        exptimes = np.unique([im.exptime for im in self.images if im.type != 'BIAS'])
        exptimes = exptimes[exptimes > threshold]
        return exptimes.tolist()

    def filter(self, object_contains=None, object_exact=None, filename_contains=None, filename_exact=None,
               image_type=None, exptime=None, limit=None):
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
            if image_type is not None and image_type != im.type:
                mask[k] = False
                continue
            if exptime is not None and exptime != im.exptime:
                mask[k] = False
                continue
        # Apply mask
        images = np.array(self.images)[mask].tolist()
        # Apply limit
        images = apply_limit(images, limit)
        # Return new ImageList
        return ImageList(images)

    def combine(self, method='median', **kwargs):
        print(f'Combining {len(self)} images using method "{method}".')
        if method == 'median':
            combine_function = median_combine
        elif method == 'mean':
            combine_function = mean_combine
        else:
            raise ValueError(f'Unknown method "{method}"!')
        result = self.image_class.combine(self.images, combine_function, **kwargs)
        print('Combine done!')
        return result
