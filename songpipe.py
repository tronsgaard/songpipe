from os.path import basename, splitext, dirname, exists, join, relpath
from os import makedirs
from glob import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyreduce import echelle
from pyreduce.reduce import ScienceExtraction, Flat, OrderTracing, BackgroundScatter, NormalizeFlatField, WavelengthCalibrationFinalize

# Settings (to be moved to a separate file eventually)
OBJECT_IN_FILENAME = False


"""
HELPER FUNCTIONS
"""
__version__ = "0.1.0"

def construct_filename(orig_filename, object=None, prepared=False, extracted=False, mode=None,
                       prefix=None, suffix=None, ext=None):
    """Construct a standardized filename based on the properties of the file"""
    filename, old_ext = splitext(orig_filename)
    if object is not None:
        filename += '_' + object.replace(' ', '_')
    # Prepend stuff
    if prefix is not None:
        filename = prefix + '_' + filename
    # Append stuff
    if mode is not None:
        filename += '_' + mode
    if prepared:
        filename += '_prep'
    elif extracted:
        filename += '_extr'
    if suffix is not None:
        filename += '_' + suffix
    if ext == None:
        if old_ext != '':
            ext = old_ext
        else:
            ext = '.fits'
    return filename + ext


def header_insert(hdr, key, value=None, comment=''):
    """Keep the header organized by grouping all pipeline keywords in a section"""
    hdr = hdr.__copy__()
    SECTION_HEADER = ('---PL---', '----PIPELINE----', '-------------------------------------')
    SECTION_FOOTER = ('--------', '----------------', '-------------------------------------')
    get_keys = lambda: list(hdr.keys())  # Get updated list of keys from header
    try:
        start = get_keys().index(SECTION_HEADER[0])
    except ValueError:
        hdr.set(*SECTION_HEADER)
        start = get_keys().index(SECTION_HEADER[0])
        hdr.insert(start, SECTION_FOOTER, after=True)
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
    if key in get_keys():
        hdr.set(key, value, comment)
    else:
        hdr.insert(end, (key, value, comment), after=True)
    return hdr

def sanitize_header(hdr):
    """Remove duplicate key/value pairs and warn of duplicate keys with different values"""
    
    keys = list(hdr.keys())
    values = list(hdr.values())
    comments = list(hdr.comments)

    new_header = fits.Header()
    count_discarded = 0
    for i in range(len(hdr)):
        key, value, comment = keys[i], values[i], comments[i]
        
        if key not in new_header: 
            # If key not already in header
            new_header.append((key, value, comment))

        elif key in new_header and value != new_header[key]:
            # If key already in header, but different value (warn the user)
            print(f'Conflicting header values: {key}: "{value}" vs. "{new_header[key]}"')
            new_header.append((key, value, comment))

        else:
            # Otherwise, don't add to header (key/value pair matches existing)
            count_discarded += 1
    print(f'{count_discarded} key/value pairs removed.')
    return new_header
         

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


def mean_combine(images):
    """Mean combine a list of 2D images"""
    data = [im.data for im in tqdm(images)]
    return Image(data=np.mean(data, axis=0))

"""
METHODS FOR PLOTTING PYREDUCE OUTPUT
"""
def plot_order_trace(flat, orders, column_range, savename=None):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()

    vmin = np.quantile(flat, .001)
    vmax = np.quantile(flat, .85)
    ax.imshow(flat, vmin=vmin, vmax=vmax, cmap='viridis')
    ax.invert_yaxis()

    nord = len(orders)
    for i in range(nord):
        x = np.arange(*column_range[i])
        ycen = np.polyval(orders[i], x).astype(int)
        ax.plot(x, ycen, linestyle='--', color='magenta', linewidth=.5)

    # Save figure to file
    if savename is not None:
        fig.savefig(savename, dpi=150)


"""
CLASS DEFINITIONS
"""


class CalibrationSet():
    """This class represents a calibration set for a single-fiber setup (e.g. F1)"""
    
    def __init__(self, images, output_dir, config, mask, instrument, mode, order_range=None, skip_existing=True):
        """Prepare to reduce calibrations for the selected mode using the selected files

        Parameters
        ----------
        images : songpipe.ImageList
            Images to use for calibration (e.g. full night including other modes and science images)
        output_dir : str
            Calibration output directory
        instrument : str
            instrument used for observations
        mode : str
            instrument mode used (e.g. "F1" or "F2")
        config : dict
            numeric reduction specific settings, like pixel threshold, which may change between runs
        skip_existing : bool
            Whether to skip reductions with existing output
        """ 
        self.images = images
        self.output_dir = output_dir
        self.config = config
        self.mask = mask
        self.instrument = instrument
        self.mode = mode
        self.order_range = order_range
        self.skip_existing = skip_existing
        self.bias = None  # TODO: Change this?

        self.data = {}
        self.data['bias'] = (None, None)
        self.data['curvature'] = (None, None)
        self.steps = {}

        self.wavelength_calibs = []

    @property
    def step_args(self):
        # (instrument, mode, target, night, output_dir, order_range)
        return (self.instrument, self.mode, None, None, self.output_dir, self.order_range)

    def combine_flats(self):
        """Combine flats"""
        step_flat = Flat(*self.step_args, **self.config['flat'])
        if exists(step_flat.savefile):
            flat, flat_header = step_flat.load(self.mask)
            print(f"Loaded existing master flat ({self.mode}) from {relpath(step_flat.savefile, self.output_dir)}.")
        else:
            print(f'Assembling master flat ({self.mode})...')
            flats = self.images.filter(image_type='FLAT', mode=self.mode)
            flats.list()
            flat, flat_header = step_flat.run(flats.files, None, self.mask)
        self.data['flat'] = (flat, flat_header)
        self.steps['flat'] = step_flat

    def trace_orders(self):
        """Trace orders in single-fiber modes"""
        try:
            return self.data['orders']
        except KeyError:
            step_orders = OrderTracing(*self.step_args, **self.config['orders'])
            if exists(step_orders.savefile):
                orders, column_range = step_orders.load()
                print(f"Loaded {len(orders)} order traces ({self.mode}) from {relpath(step_orders.savefile, self.output_dir)}.")
            else:
                print(f'Tracing orders in ({self.mode})...')
                step_flat = self.steps['flat']
                orders, column_range = step_orders.run([step_flat.savefile], self.mask, None)
                print(f"Traced {len(orders)} orders in image {relpath(step_flat.savefile, self.output_dir)}.")
            self.data['orders'] = (orders, column_range)
            self.steps['orders'] = step_orders
            self.plot_trace()
            return (orders, column_range)

    def plot_trace(self):
        """Custom order plot"""
        #name, _ = splitext(self.steps['orders'].savefile)
        savename = join(self.output_dir, f"plot_trace_{self.mode}.png")
        if self.skip_existing and exists(savename):
            print(f'Order plot already exists: {relpath(savename, self.output_dir)}')
        else:
            flat, fhead = self.data['flat']
            orders, column_range = self.data['orders']
            plot_order_trace(flat, orders, column_range, savename=savename)  # Custom plot routine
            print(f'Order plot saved to: {relpath(savename, self.output_dir)}')

    
    def measure_scattered_light(self):
        print(f'Measuring scattered light in master flat...')
        step_scatter = BackgroundScatter(*self.step_args, **self.config['scatter'])
        if exists(step_scatter.savefile):
            scatter = step_scatter.load()
            print('Loaded existing scattered light fit for image {}')
        else:
            print('Measuring scattered light')
            scatter = step_scatter.run([self.steps['flat'].savefile], self.mask, None, self.data['orders'])
        self.data['scatter'] = scatter
        self.steps['scatter'] = step_scatter

    def measure_curvature(self):
        # TODO (not needed with fiber)
        self.data['curvature'] = (None, None)

    def normalize_flat(self):
        # Norm flat and blaze
        step_normflat = NormalizeFlatField(*self.step_args, **self.config['norm_flat'])
        try:
            norm, blaze = step_normflat.load()
            print(f'Loaded existing normflat ({self.mode}) from {relpath(step_normflat.savefile, self.output_dir)}')
        except FileNotFoundError:
            print(f'Normalizing {self.mode} flat field...')
            norm, blaze = step_normflat.run(self.data['flat'], self.data['orders'], self.data['scatter'], self.data['curvature'])
        self.data['norm_flat'] = (norm, blaze)
        self.steps['norm_flat'] = step_normflat

    def extract(self, image, savedir=None, skip_existing=True, wave=None):
        step_science = ScienceExtraction(*self.step_args, **self.config['science'])
        self.steps['science'] = step_science
        orig_filename = basename(image.filename)
        print(f'Working on file: {orig_filename}')
        if self.check_extracted_exists(orig_filename) and skip_existing is True:
            print(f'Extracted spectrum already exists for {orig_filename}')
            # FIXME: Check if wavelength solution needs to be appended
            return self.load_extracted(orig_filename, savedir=savedir)
        else:
            im, head = step_science.calibrate([image.filename], self.mask, self.data['bias'], self.data['norm_flat'])
            spec, sigma, _, column_range = step_science.extract(im, head, self.data['orders'], self.data['curvature'])  # TODO: scattered light as kw arg
            # TODO: Make diagnostic plot
            return self.save_extracted(orig_filename, head, spec, sigma, column_range, savedir=savedir)
            
    def save_extracted(self, orig_filename, head, spec, sigma, column_range, savedir=None):
        nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode)
        head = header_insert(head, 'PL_MODE', self.mode, 'Observing mode (e.g. fiber)')
        try:
            wave = self.wavelength_calibs[0]
        except IndexError:
            wave = None
        print(f'Saving spectrum to file: {nameout}')
        echelle.save(nameout, head, spec=spec, sig=sigma, wave=wave, columns=column_range)
        return [Spectrum(filename=nameout)]
    
    def load_extracted(self, orig_filename, savedir=None):
        nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode)
        return [Spectrum(filename=nameout)]

    def get_extracted_filename(self, orig_filename, savedir=None, mode=None):
        orig_filename = basename(orig_filename)
        orig_filename, _ = splitext(orig_filename)
        orig_filename = orig_filename.replace('_prep', '')
        new_filename = construct_filename(orig_filename, extracted=True, mode=mode)
        if savedir is None:
            savedir = self.output_dir
        return join(savedir, new_filename)
    
    def check_extracted_exists(self, orig_filename, savedir=None):
        return exists(self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode))


class MultiFiberCalibrationSet(CalibrationSet):
    """This class represents a calibration set for a dual-fiber setup (e.g. F12)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_mode_calibs = {}

    @property
    def sub_modes(self):
        return list(self.sub_mode_calibs.keys())

    def link_single_fiber_calibs(self, *args):
        """Link the calibration sets for sub modes (e.g. F1 and F2)"""
        for calibration_set in args:
            mode = calibration_set.mode
            self.sub_mode_calibs[mode] = calibration_set

    def trace_orders(self):
        """Combine order trace for multi-fiber mode"""

        # Fetch single-fiber traces
        orders, column_range, order_modes = [], [], []
        for mode, calibs in self.sub_mode_calibs.items():
            sub_orders, sub_column_range = calibs.trace_orders()  # Get it if already traced, otherwise run trace on the other fiber
            orders.append(sub_orders)
            column_range.append(sub_column_range)
            order_modes += [mode]*len(sub_orders)  # Append a list of e.g. ['F1', 'F1', ..., 'F1', 'F1']

        # Combine
        orders = np.vstack(orders)
        column_range = np.vstack(column_range)
        order_modes = np.vstack(order_modes)
        
        # Sort orders by detector position
        xmin = np.min([cr[0] for cr in column_range])
        xmax = np.max([cr[1] for cr in column_range])
        xmid = (xmax - xmin) // 2
        ycen = [np.polyval(orders[i], xmid).astype(int) for i in range(len(orders))]

        subs = np.argsort(ycen)
        orders = orders[subs]
        column_range = column_range[subs]
        order_modes = order_modes[subs]
        
        # Save for later
        self.data['orders'] = (orders, column_range)
        self.order_modes = order_modes.flatten()
        self.plot_trace()

    def save_extracted(self, orig_filename, head, spec, sigma, column_range, savedir=None):
        """Ensure that each fiber is saved to a separate file"""

        # Split extracted orders into modes/fibers
        results = []
        for sub_mode in self.sub_modes:
            selected = self.order_modes == sub_mode  # E.g. all orders from F1
            nord = np.sum(selected)

            # Save this mode
            nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=sub_mode)
            head = header_insert(head, 'PL_MODE', sub_mode, 'Observing mode (e.g. fiber)')
            try:
                whead, wave = self.sub_mode_calibs[sub_mode].wavelength_calibs[0]
            except IndexError:
                whead, wave = None, None
            print(f'Saving {nord} orders from mode {sub_mode} to file: {nameout}')
            echelle.save(nameout, head, spec=spec[selected], sig=sigma[selected], columns=column_range[selected], wave=wave)
            results.append(Spectrum(filename=nameout))
        return results
    
    def load_extracted(self, orig_filename, savedir=None):
        results = []
        for sub_mode in self.sub_modes:
            nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=sub_mode)
            results.append(Spectrum(filename=nameout))
        return results
    
    def check_extracted_exists(self, orig_filename, savedir=None):
        for mode, calib in self.sub_mode_calibs.items():
            if calib.check_extracted_exists(orig_filename, savedir=savedir) is False:
                return False
        return True



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
    def type(self):
        return self.get_header_value('IMAGETYP')

    @property
    def mode(self):
        """Returns the instrument mode, currently (MtKent): F1, F2, or F12, SLIT, DARK, UNKNOWN"""
        try:
            # First look for SONGPIPE header keyword
            return self.get_header_value('PL_MODE')
        except KeyError:
            # Otherwise, derive from rest of header
            if self.type in ('DARK', 'BIAS'):
                return 'DARK'
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
            # In any other case (including all observations):
            return 'UNKNOWN'  # FIXME: Figure out a way to detect if there is light in both fibre

    # Misc
    def construct_filename(self, **kwargs):
        if 'object' in kwargs:
            obj = kwargs.pop('object')  # Removes item from kwargs
        elif OBJECT_IN_FILENAME is True:
            obj = self.object
        else:
            obj = None
        return construct_filename(basename(self.get_header_value('FILE')), object=obj, **kwargs)

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


class HighLowImage(Image):
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
    def combine(cls, images, combine_function):
        """Combine a list of HighLowImages into one HighLowImage"""
        print('Combining high gain images...')
        high_gain_image = combine_function([im.high_gain_image for im in images])
        print('Combining low gain images...')
        low_gain_image = combine_function([im.low_gain_image for im in images])
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


class FrameList:
    pass

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
               image_type=None, mode=None, exptime=None, limit=None):
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
                if isinstance(image_type, str) or not hasattr(image_type, '__iter__'):
                    image_type = [image_type]
                if im.type not in image_type:
                    mask[k] = False
                    continue
            if mode is not None:
                if isinstance(mode, str) or not hasattr(mode, '__iter__'):
                    mode = [mode]
                if im.mode not in mode:
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


class SpectrumList(FrameList):
    pass
