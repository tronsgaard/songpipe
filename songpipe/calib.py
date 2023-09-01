from os.path import basename, splitext, exists, join, relpath, dirname
from os import makedirs
import numpy as np
from pyreduce import echelle
from pyreduce.reduce import ScienceExtraction, Flat, OrderTracing, BackgroundScatter, NormalizeFlatField

from .plotting import plot_order_trace
from .misc import construct_filename, header_insert
from .spectrum import Spectrum

"""
This module contains the `CalibrationSet` class, and the subclass `MultiFiberCalibrationSet`, 
with methods that wrap the functionality of PyReduce.
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
            print(f'Loaded existing scattered light fit from {relpath(step_scatter.savefile, self.output_dir)}')
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
        if exists(step_normflat.savefile):
            norm, blaze = step_normflat.load()
            print(f'Loaded existing normflat ({self.mode}) from {relpath(step_normflat.savefile, self.output_dir)}')
        else:
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
        makedirs(dirname(nameout), exist_ok=True)
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
