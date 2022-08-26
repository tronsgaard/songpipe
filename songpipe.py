from os.path import basename, splitext, dirname, exists
from os import makedirs
from glob import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    if hdr is None:
        return
    PIPELINE_HEADER = ('---PL---', '----PIPELINE----', '-------------------------------------')
    get_keys = lambda: list(hdr.keys())  # Get updated list of keys from header
    try:
        start = get_keys().index(PIPELINE_HEADER[0])
    except ValueError:
        hdr.set(*PIPELINE_HEADER)
        start = get_keys().index(PIPELINE_HEADER[0])
    end = start

    # Determine end of section
    in_section = True
    keys = get_keys()
    while in_section is True:
        if end+1 >= len(keys) or keys[end+1][0] == '-' or keys[end+1] == 'COMMENT':
            in_section = False
        else:
            end += 1

    # Insert header key/value
    hdr.insert(end, (key, value, comment), after=True)

def apply_limit(array, limit):
    """SQL-like limit syntax"""
    if not hasattr(limit, '__iter__'):
        limit = (limit,)
    return array[slice(*limit)]


def median_combine(images):
    """Median combine a list of 2D images"""
    data = [im.data for im in tqdm(images)]
    return Image(data=np.median(data, axis=0))

def mean_combine(images):
    """Mean combine a list of 2D images"""
    data = [im.data for im in tqdm(images)]
    return Image(data=np.mean(data, axis=0))


class ImageBase:
    def __init__(self):
        self._header = None
        self._data = None
        self.filename = None
        self.ext = None
        self.bias_subtracted = False
        self.dark_subtracted = False
        self.oriented = False
        self.gain_applied = False

    @classmethod
    def from_file(cls, filename, **kwargs):
        raise NotImplementedError

    @classmethod
    def combine(cls, images, combine_function):
        raise NotImplementedError

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        raise NotImplementedError

    @property
    def object(self):
        return self.header['OBJECT']

    @property
    def exptime(self):
        return self.header['EXPTIME']

    @property
    def type(self):
        return self.header['IMAGETYP']

    def construct_filename(self, **kwargs):
        return construct_filename(basename(self.header['FILE']), object=self.object, **kwargs)

    def show(self, ax=None, vmin=None, vmax=None):
        assert self.data is not None
        if ax is None:
            plt.figure(figsize=(7,5))
        else:
            plt.sca(ax)
        plt.imshow(self.data, vmin=vmin, vmax=vmax)

    def hist(self, ax=None, bins=100, **kwargs):
        assert self.data is not None
        if ax is None:
            plt.figure(figsize=(7,5))
        else:
            plt.sca(ax)
        plt.hist(self.data.flatten(), bins=bins, **kwargs)

    def clear_data(self):
        self._data = None

    def make_hdulist(self):
        raise NotImplementedError

    def save_fits(self, out_filename, overwrite=False):
        """Save image to a FITS file"""
        hdu = self.make_hdulist()
        print(f'Saving to {out_filename}...')
        makedirs(dirname(out_filename), exist_ok=True)  # Ensure that output folder exists
        hdu.writeto(out_filename, overwrite=overwrite)

    # Transformations
    def subtract_bias(self, bias):
        raise NotImplementedError

    def subtract_dark(self, dark):
        raise NotImplementedError

    def orient(self, rotation=0, flip_updown=False, flip_leftright=False):
        raise NotImplementedError

    def apply_gain(self, **kwargs):
        raise NotImplementedError

    def calculate_variance(self, **kwargs):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError


class Image(ImageBase):
    def __init__(self, header=None, data=None, filename=None, ext=0,
                 bias_subtracted=False, dark_subtracted=False):
        super().__init__()
        # Load image
        self._header = header
        self._data = data
        self.filename = filename
        self.ext = ext
        # Get transformations from fits header
        if self._header is not None:
            self.bias_subtracted = self._header.get('PL_BISUB', bias_subtracted)
            self.dark_subtracted = self._header.get('PL_DASUB', dark_subtracted)

    @classmethod
    def from_file(cls, filename, ext=0, load_data=False):
        with fits.open(filename) as h:
            hdu = h[ext]
            if load_data:
                print(f'Loading FITS data from "{filename}" (ext {ext})')
                return cls(header=hdu.header, data=hdu.data, filename=filename, ext=ext)
            else:
                return cls(header=hdu.header, filename=filename, ext=ext)

    @classmethod
    def combine(cls, images, combine_function):
        return combine_function(images)  # Returns an image

    @property
    def data(self):
        if self._data is not None:
            return self._data
        if self.filename is not None:
            with fits.open(self.filename) as h:
                print(f'Loading FITS data from "{self.filename}" (ext {self.ext})')
                self._data = h[self.ext].data
                return self._data

    def make_hdulist(self):
        return fits.HDUList([fits.PrimaryHDU(data=self.data, header=self.header)])

    # Transformations
    def subtract_bias(self, bias):
        assert self.bias_subtracted is False
        self._data = self.data - bias.data
        header_insert(self._header, 'PL_BISUB', True, 'Bias subtracted')
        self.bias_subtracted = True

    def subtract_dark(self, dark):
        assert self.bias_subtracted
        assert self.dark_subtracted is False
        self._data = self.data - dark.data
        header_insert(self._header, 'PL_DASUB', True, 'Dark subtracted')
        self.dark_subtracted = True

    def orient(self, rotation=0, flip_updown=False, flip_leftright=False):
        if flip_updown:
            self._data = np.flipud(self.data)
        if flip_leftright:
            self._data = np.fliplr(self.data)
        if rotation != 0:
            self._data = np.rot90(self.data, k=rotation//90)
        header_insert(self._header, 'PL_ORIEN', True, 'Oriented')
        self.oriented = True

    def apply_gain(self, gain_factor):
        assert self.bias_subtracted
        assert self.dark_subtracted
        assert self.gain_applied is False
        self._data *= gain_factor
        self.gain_applied = True

    def calculate_variance(self, **kwargs):
        raise NotImplementedError

    def __add__(self, other):
        data = self.data + other.data
        return Image(data=data)


class HighLowImage(ImageBase):
    def __init__(self, high_gain_image, low_gain_image, filename=None):
        super().__init__()
        self.high_gain_image = high_gain_image
        self.low_gain_image = low_gain_image
        self.filename = filename
        self.bias_subtracted = self.high_gain_image.bias_subtracted
        self.dark_subtracted = self.high_gain_image.dark_subtracted

    @classmethod
    def from_file(cls, filename):
        # Load image
        high_gain_image = Image.from_file(filename=filename, ext=0)
        low_gain_image  = Image.from_file(filename=filename, ext=1)
        return cls(high_gain_image, low_gain_image, filename=filename)

    @classmethod
    def combine(cls, images, combine_function):
        high_gain_image = combine_function([im.high_gain_image for im in images])
        low_gain_image = combine_function([im.high_gain_image for im in images])
        return HighLowImage(high_gain_image, low_gain_image)  # Return a HighLowImage

    def _dual_plot(self, func_high, func_low, **kwargs):
        fig, ax = plt.subplots(ncols=2)
        fig.set_size_inches(12,5)
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

    def clear_data(self):
        self.high_gain_image.clear_data()
        self.low_gain_image.clear_data()

    def make_hdulist(self):
        return fits.HDUList([
            fits.PrimaryHDU(data=self.high_gain_image.data, header=self.high_gain_image.header),
            fits.ImageHDU(data=self.high_gain_image.data, header=self.high_gain_image.header),
        ])

    @property
    def header(self):
        return self.high_gain_image.header

    # Transformations
    def subtract_bias(self, bias):
        assert self.bias_subtracted is False
        self.high_gain_image.subtract_bias(bias.high_gain_image)
        self.low_gain_image.subtract_bias(bias.low_gain_image)
        self.bias_subtracted = True

    def subtract_dark(self, dark):
        assert self.bias_subtracted
        assert self.dark_subtracted is False
        self.high_gain_image.subtract_dark(dark.high_gain_image)
        self.low_gain_image.subtract_dark(dark.low_gain_image)
        self.dark_subtracted = True

    def apply_gain(self, gain_high=0.78, gain_low=15.64):
        # electrons/ADU for HIGHGAIN and LOWGAIN image, respectively: [0.78, 15.64]
        assert self.bias_subtracted
        assert self.dark_subtracted
        assert self.gain_applied is False
        self.high_gain_image.apply_gain(gain_high)
        self.low_gain_image.apply_gain(gain_low)
        self.gain_applied = True

    def orient(self, rotation=0, flip_updown=False, flip_leftright=False):
        self.high_gain_image.orient(rotation=rotation, flip_updown=flip_updown, flip_leftright=flip_leftright)
        self.low_gain_image.orient(rotation=rotation, flip_updown=flip_updown, flip_leftright=flip_leftright)
        self.oriented = True

    def calculate_variance(self):
        raise NotImplementedError

    def merge_high_low(self, threshold=3000):
        assert self.bias_subtracted
        assert self.dark_subtracted
        assert self.gain_applied

        high_gain_data = self.high_gain_image.data
        low_gain_data = self.low_gain_image.data

        mask = high_gain_data >= threshold  # TODO: What about saturated pixels?
        merged = high_gain_data.copy()
        merged[mask] = low_gain_data[mask]

        return Image(data=merged, header=self.header)

    def __add__(self, other):
        high = self.high_gain_image + other.high_gain_image
        low = self.low_gain_image + other.low_gain_image
        return HighLowImage(high, low)


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
        images = [image_class.from_file(f) for f in tqdm(files)]
        return ImageList(images)

    @classmethod
    def from_filemask(cls, filemask, image_class=Image, limit=None):
        return cls.from_files(glob(filemask), image_class=image_class, limit=limit)

    def __len__(self):
        """Provides len() compatibility"""
        return len(self.images)

    def list(self):
        """Print a pretty list of filenames and some fits keywords"""
        if len(self) == 0:
            return

        buffer = []
        widths = {}  # Column widths
        for im in self.images:
            d = {}
            if im.filename is not None:
                d['filename'] = basename(im.filename)
            else:
                d['filename'] = ''
            d['image_type'] = im.type
            d['exptime'] = im.exptime
            d['object'] = im.object
            buffer.append(d)
            # Update column widths
            for k in d.keys():
                widths[k] = max((len(str(d[k])), widths.get(k, 0),))
        # Define format string
        fmt = '  '.join([f'{{{k}:<{w}}}' for k,w in widths.items()])
        # Print headers
        headers = {k:k.title() for k in widths.keys()}
        headers['image_type'] = 'Type'
        headers['exptime'] = 'Exp'
        print(fmt.format(**headers))
        # Print table rows
        for d in buffer:
            print(fmt.format(**d))

    def summary(self):
        """Print summary of image types in the list"""
        count = {}
        for im in self.images:
            count[im.type] = count.get(im.type, 0) + 1

        for k,n in count.items():
            print(k, n)

    def get_exptimes(self, threshold=0.1):
        """Return a list of exptimes"""
        exptimes = np.unique([im.exptime for im in self.images if im.type != 'BIAS'])
        exptimes = exptimes[exptimes > threshold]
        return exptimes.tolist()

    def filter(self, object_contains=None, object_exact=None, image_type=None, exptime=None, limit=None):
        mask = [True]*len(self)
        for k, im in enumerate(self.images):
            if object_contains is not None and object_contains not in im.object:
                mask[k] = False
                continue
            if object_exact is not None and object_exact != im.object:
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

    def combine(self, method='median'):
        print(f'Combining {len(self)} images using method "{method}".')
        if method == 'median':
            combine_function = median_combine
        elif method == 'mean':
            combine_function = mean_combine
        else:
            raise ValueError(f'Unknown method "{method}"!')
        result = self.image_class.combine(self.images, combine_function)
        print('Combine done!')
        return result
