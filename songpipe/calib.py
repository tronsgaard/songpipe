"""
This module contains the `CalibrationSet` class, and the subclass `MultiFiberCalibrationSet`, 
with methods that wrap the functionality of PyReduce.
"""
from os.path import basename, splitext, exists, join, relpath, dirname
from os import makedirs
import numpy as np
from pyreduce import echelle
from pyreduce.reduce import ScienceExtraction, Flat, OrderTracing, BackgroundScatter, \
    NormalizeFlatField, WavelengthCalibrationMaster, WavelengthCalibrationFinalize
from pyreduce.extract import fix_parameters
from pyreduce.wavelength_calibration import LineList
from pyreduce import __version__ as pyreduce_version

from .plotting import plot_order_trace
from .misc import construct_filename, header_insert
from .image import Image
from .spectrum import Spectrum, SpectrumList
from . import __version__ as songpipe_version

from logging import getLogger
logger = getLogger(__name__)

class CalibrationSet():
    """This class represents a calibration set for a single-fiber setup (e.g. F1)"""
    
    def __init__(self, images, output_dir, config, mask, instrument, mode, 
                 order_range=None, skip_existing=True, mode_in_extracted_filenames=True):
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
        mode_in_extracted_filenames : bool
            Include mode (slit or fiber) in extracted filenames
        """ 
        self.images = images
        self.output_dir = output_dir
        self.config = config
        self.mask = mask
        self.instrument = instrument
        self.mode = mode
        self.order_range = order_range
        self.skip_existing = skip_existing
        self.mode_in_extracted_filenames = mode_in_extracted_filenames
        self.bias = None  # TODO: Change this?

        self.data = {}
        self.data['bias'] = (None, None)
        self.data['curvature'] = (None, None)
        self.steps = {}

    @property
    def step_args(self):
        # (instrument, mode, target, night, output_dir, order_range)
        return (self.instrument, self.mode, None, None, self.output_dir, self.order_range)
    
    @property
    def wavelength_calibs(self):
        """Ensure that self.wavelength_calibs is never None or undefined"""
        try:
            assert isinstance(self._wavelength_calibs, SpectrumList)
        except (AssertionError, AttributeError):  # wrong type or undefined
            self._wavelength_calibs = SpectrumList([])
        return self._wavelength_calibs
    
    @wavelength_calibs.setter
    def wavelength_calibs(self, value):
        """Ensure that self.wavelength_calibs is always a SpectrumList"""
        if not isinstance(value, SpectrumList):
            raise TypeError('wavelength_calibs must be a SpectrumList')
        self._wavelength_calibs = value

    def combine_flats(self, min_flat_images=1):
        """Combine flats"""
        step_flat = Flat(*self.step_args, **self.config['flat'])
        if exists(step_flat.savefile):
            logger.info(f"Loading existing master flat ({self.mode}) from {relpath(step_flat.savefile, self.output_dir)}.")
            flat, flat_header = step_flat.load(self.mask)
        else:
            logger.info(f'Assembling master flat ({self.mode})...')
            flats = self.images.filter(image_type='FLAT', mode=self.mode)
            flats.list()
            if len(flats) < min_flat_images:
                raise NotEnoughFlatsError(f'Not enough flat images. Expected {min_flat_images}, found {len(flats)}')
            flat, flat_header = step_flat.run(flats.files, None, self.mask)
        self.data['flat'] = (flat, flat_header)
        self.steps['flat'] = step_flat

    def trace_orders(self, ymin=None, ymax=None, min_width=0.4, ref_column=None, target_nord=None, overwrite_plot=False):
        """
        Trace orders in single-aperture mode
        ymin,ymax is used for trimming extreme orders, pixels refer to central column (or ref_column)
        target_nord is the expected number of orders
        """
        try:
            return self.data['orders']
        except KeyError:
            step_orders = OrderTracing(*self.step_args, **self.config['orders'])
            if exists(step_orders.savefile):
                orders, column_range = step_orders.load()
                logger.info(f"Loaded {len(orders)} order traces ({self.mode}) from {relpath(step_orders.savefile, self.output_dir)}.")
            else:
                logger.info(f'Tracing orders in ({self.mode})...')
                step_flat = self.steps['flat']
                orders, column_range = step_orders.run([step_flat.savefile], self.mask, None)
                logger.info(f"Traced {len(orders)} orders in image {relpath(step_flat.savefile, self.output_dir)}.")

            # Trim extreme orders
            # First, remove orders that span less than a given fraction of the detector width
            ysize, xsize = self.images[0].shape
            order_width = column_range[:,1] - column_range[:,0]
            ok = order_width / xsize > min_width

            # Evaluate each order trace polynomial in a central column and compare with ymin, ymax
            if ref_column is None:
                ref_column = (np.max(column_range) - np.min(column_range)) // 2
            ypos = np.hstack([np.polyval(orders[i], [ref_column]) for i in range(len(orders))])
            if ymin is not None:
                ok = ok & (ypos >= ymin)
            if ymax is not None:
                if ymax < 0:  # Support negative indexing, -1 is last pixel
                    ymax += ysize
                ok = ok & (ypos <= ymax)

            # Remove orders from (orders,column_range) tuple 
            if np.sum(ok==False) > 0:
                logger.info(f'Trimming orders: {np.where(ok==False)[0].tolist()}')
                orders, column_range = (orders[ok], column_range[ok])
                
                # override order tracing file
                np.savez(step_orders.savefile, orders=orders, column_range=column_range)
                logger.info("Updated order tracing file: %s", step_orders.savefile)
                overwrite_plot = True

            # Verify that we found the expected number of orders between ymin and ymax
            if target_nord is not None:
                try:
                    assert ((nord := len(orders)) == target_nord)
                except AssertionError:
                    logger.warning(f'Unexpected number of orders found: {nord}. Expected: {target_nord}.')

            self.data['orders'] = (orders, column_range)
            self.steps['orders'] = step_orders
            self.plot_trace(overwrite=overwrite_plot)
            return (orders, column_range)

    def plot_trace(self, overwrite=False,**kwargs):
        """Custom order plot"""
        #name, _ = splitext(self.steps['orders'].savefile)
        savename = join(self.output_dir, f"plot_trace_{self.mode}.png")
        if self.skip_existing and exists(savename) and overwrite is False:
            logger.info(f'Order plot already exists: {relpath(savename, self.output_dir)}')
        else:
            flat, fhead = self.data['flat']
            orders, column_range = self.data['orders']
            widths = self.get_extraction_widths()
            plot_order_trace(flat, orders, column_range, widths=widths, savename=savename, **kwargs)  # Custom plot routine
            logger.info(f'Order plot saved to: {relpath(savename, self.output_dir)}')

    def get_extraction_widths(self, xwd=None, step_name='science', image_shape=None):
        """
        Converts fractional extraction width into pixels for each order.
        Use argument xwd if supplied, otherwise get from config[step_name]
        """
        try:
            orders, cr = self.data['orders']
            nord = len(orders)
        except KeyError:
            raise Exception('Cannot calculate extraction widths without order trace')
        xwd = self.config[step_name]['extraction_width']
        if image_shape is None:
            image_shape = self.images[0].shape
        nrow, ncol = image_shape
        xwd, cr, orders = fix_parameters(xwd, cr, orders, nrow, ncol, nord, 
                                         ignore_column_range=True)
        return xwd
    
    def log_extraction_widths(self, **kwargs):
        """Calculate extraction widths and print/log for each order"""
        xwd = self.get_extraction_widths(**kwargs)
        orders, cr = self.data['orders']
        if hasattr(self, 'order_modes'):
            order_modes = self.order_modes
        else:
            order_modes = [self.mode] * len(orders)
        # Loop over orders
        logger.info(f'{self.mode} extraction widths:')
        for i in range(len(orders)):
            x0, x1 = xwd[i]
            c0, c1 = cr[i]
            om = order_modes[i]
            logger.info(f'Order {i} ({om}): '
                        f'{x0}px (below) / {x1}px (above) '
                        f'| column range: [{c0};{c1}]')

    
    def measure_scattered_light(self):
        logger.info(f'Measuring scattered light in master flat...')
        step_scatter = BackgroundScatter(*self.step_args, **self.config['scatter'])
        if exists(step_scatter.savefile):
            logger.info(f'Loading existing scattered light fit from {relpath(step_scatter.savefile, self.output_dir)}')
            scatter = step_scatter.load()
        else:
            logger.info('Measuring scattered light')
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
            logger.info(f'Loading existing normflat ({self.mode}) from {relpath(step_normflat.savefile, self.output_dir)}')
            norm, blaze = step_normflat.load()
        else:
            logger.info(f'Normalizing {self.mode} flat field...')
            norm, blaze = step_normflat.run(self.data['flat'], self.data['orders'], self.data['scatter'], self.data['curvature'])
        # Save normflat as fits
        fname, _ = splitext(step_normflat.savefile)
        fname += '.fits'
        if exists(fname):
            logger.info('Normflat already written to FITS file. Skipping..')
        else:
            logger.info(f'Writing normflat to FITS file: {fname}')
            from astropy.io.fits import writeto
            writeto(filename=fname, data=norm, overwrite=True)
        self.data['norm_flat'] = (norm, blaze)
        self.steps['norm_flat'] = step_normflat

    def extract_flat(self):
        """Extract the summed flat as a spectrum"""
        flat, flat_header = self.data['flat']
        step_flat = self.steps['flat']
        filename = step_flat.savefile
        flat_image = Image(header=flat_header, data=flat, filename=filename)
        logger.info(f'Extracting summed flat ({self.mode})')
        return self.extract(flat_image, savedir=self.output_dir)

    def extract(self, image, savedir=None, skip_existing=True):
        """
        Extract the spectrum and return as <Spectrum> object. 
        If skip_existing=True, load existing spectrum and return as <Spectrum> object.
        """
        orig_filename = basename(image.filename)
        if self.check_extracted_exists(orig_filename, savedir=savedir) and skip_existing is True:
            logger.info(f'Extracted spectrum already exists for {orig_filename}')
            # FIXME: Check if wavelength solution needs to be appended
            return self.load_extracted(orig_filename, savedir=savedir)
        else:
            logger.info(f'Working on file: {orig_filename}')
            # Using step ScienceExtraction for both science and wave - only config changes
            if image.type == 'THAR':
                config = self.config['wavecal_master']
            else:
                config = self.config['science']
            step_science = ScienceExtraction(*self.step_args, **config)
            im, head = step_science.calibrate([image.filename], self.mask, self.data['bias'], self.data['norm_flat'])
            spec, sigma, _, column_range = step_science.extract(im, head, self.data['orders'], self.data['curvature']) #, scatter=self.data['scatter'])
            # Apply gain
            if image.gain_applied is False:
                gain_factor = head['e_gain']  # e-/ADU
                spec = gain_factor * spec
                head = header_insert(head, 'PL_GNAPL', True)
            # TODO: Make diagnostic plot
            return self.save_extracted(image, head, spec, sigma, column_range, savedir=savedir)
            
    def save_extracted(self, image, head, spec, sigma, column_range, savedir=None):
        """Save the extracted spectrum, assigning the nearest wavelength solution if available"""
        # Determine output filename
        orig_filename = basename(image.filename)
        nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode)
        # Add mode information to header
        head = header_insert(head, 'PL_MODE', self.mode, 'Extracted mode (fiber/slit)')
        # Specify units of the extracted spectrum
        head = header_insert(head, 'PL_UNITS', 'electrons', 'Extracted units')
        # Add songpipe and PyReduce version
        head = header_insert(head, 'PL_VERS', songpipe_version, 'SONGPIPE version')
        head = header_insert(head, 'E_VERS', pyreduce_version, 'PyReduce version')
        # Fetch blaze function
        _, cont = self.data['norm_flat']  # (norm, blaze)
        # Pick a wavelength calibration to assign
        wave = None
        if image.type != 'THAR':
            # ThAr spectra are supposed to be calibrated by their own wavelength solution
            logger.info('Finding ThAr solution closest in time..')
            try:
                # Assume that self.wavelength_calibs have already been filtered to the correct mode
                thar_spectra = self.wavelength_calibs.filter(image_type='THAR')
                thar = thar_spectra.get_closest(image.mjd_mid)  # Throws IndexError if empty
                wave = thar.wave  # Note: Could be None, if wavelength solution step was skipped
                if wave is None:
                    logger.warning(f'Closest ThAr spectrum has no wavelength solution: {relpath(thar.filename, self.output_dir)}')
                else:
                    logger.info(f'Assigned wavelength solution from {relpath(thar.filename, self.output_dir)}')
                    # Add info about wavelength solution to header
                    head = header_insert(head, 'W_FILENM', basename(thar.filename), 'Origin of wavelenth solution')
                    head = header_insert(head, 'W_MJDMID', thar.mjd_mid, 'MJD_MID of wavelength calibration')
                    head = header_insert(head, 'W_CREATS', thar.header.get('W_CREATS', 'N/A'), 'Timestamp of wavelength solution')
                    head = header_insert(head, 'W_NTHAR', thar.header.get('W_NTHAR'), 'Number of ThAr lines in calibration')
                    head = header_insert(head, 'W_RMSALL', thar.header.get('W_RMSALL'), 'RMS of wavelength solution (m/s)')
                    head = header_insert(head, 'W_MADALL', thar.header.get('W_MADALL'), 'MAD of wavelength solution (m/s)')
            except (IndexError):
                logger.warning(f'No wavelength solution could be assigned')
        # Save to file
        logger.info(f'Saving spectrum to file: {nameout}')
        makedirs(dirname(nameout), exist_ok=True)
        echelle.save(nameout, head, spec=spec, sig=sigma, wave=wave, cont=cont, columns=column_range)
        # Return as Spectrum object
        return [Spectrum(filename=nameout)]
    
    def load_extracted(self, orig_filename, savedir=None):
        nameout = self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode)
        return [Spectrum(filename=nameout)]

    def get_extracted_filename(self, orig_filename, savedir=None, mode=None):
        orig_filename = basename(orig_filename)
        orig_filename, _ = splitext(orig_filename)
        orig_filename = orig_filename.replace('_prep', '')
        if self.mode_in_extracted_filenames is False:
            mode = None
        new_filename = construct_filename(orig_filename, extracted=True, mode=mode, ext='.fits')
        if savedir is None:
            savedir = self.output_dir
        return join(savedir, new_filename)
    
    def check_extracted_exists(self, orig_filename, savedir=None):
        return exists(self.get_extracted_filename(orig_filename, savedir=savedir, mode=self.mode))
    
    def solve_wavelengths(self, linelist_path, savedir=None, skip_existing=True):
        """Solve all ThAr spectra in self.wavelength_calibs"""
        if savedir is None:
            savedir = self.output_dir

        try:
            thar_spectra = self.wavelength_calibs.filter(image_type='THAR')
            nthar = len(thar_spectra)
        except AttributeError:
            # wavelength_calibs is None
            nthar = 0

        if nthar == 0:
            logger.info(f'No ThAr spectra to solve in mode {self.mode}')
            return
        
        logger.info(f'Solving ThAr wavelengths for mode {self.mode} ({nthar} ThAr spectra)')
        logger.info(f'Linelist path: {linelist_path}')

        # Loop through ThAr spectra
        for thar in thar_spectra:
            if thar.wave is not None and skip_existing is True:
                logger.info(f'Wavelength solution already exists in file {relpath(thar.filename, self.output_dir)}')
            else:
                logger.info(f'Wavelength calibration: {relpath(thar.filename, self.output_dir)}')
                
                #step_args = calibration_set.step_args
                # (instrument, mode, target, night, output_dir, order_range)
                step_args = (self.instrument, self.mode, None, None, savedir, self.order_range)
                step_wavecal = WavelengthCalibrationFinalize(*step_args, **self.config['wavecal'])

                wavecal_master = (thar.spec, thar.header)
                linelist = LineList.load(linelist_path)
                wave, coef, linelist = step_wavecal.run(wavecal_master, linelist)

                # Add wavelength solution to Spectrum object 
                thar.wave = wave
                # Add timestamp to header
                from astropy.time import Time
                now = Time.now().isot  # Current UTC timestamp
                thar.header_insert('W_CREATS', now, 'Timestamp of wavelength solution')
                # Add number of lines to header
                lines = linelist.data[linelist['flag'] == True]  # Select lines that were used
                thar.header_insert('W_NTHAR', len(lines), 'Number of ThAr lines used')
                # Calculate RMS and MAD of wavelength solution and add to header
                residuals = lines['wlc'] - lines['wll']  # WLL = listed wavelength // WLC = fitted wavelength
                vresiduals = residuals/lines['wll'] * 299792458.
                rms = np.sqrt(np.nanmean(vresiduals**2))
                mad = np.nanmedian(np.abs(vresiduals))
                logger.info(f'Root Mean Square (RMS) of wavelength solution: {rms:.2f} m/s')
                logger.info(f'Median Absolute Deviation (MAD) of wavelength solution: {mad:.2f} m/s')
                thar.header_insert('W_RMSALL', np.round(rms, 2), 'RMS of wavelength solution (m/s)')
                thar.header_insert('W_MADALL', np.round(mad, 2), 'MAD of wavelength solution (m/s)')
                # Calculate MAD and count number of lines in each order
                calc_mad = lambda x: round(np.median(np.abs(x)), 2)
                nansafe = lambda z: z if np.isfinite(z) else str(z)  # convert nan to string before saving to FITS header
                for i in range(thar.nord):
                    mask = lines['order']==i
                    vres = vresiduals[mask]
                    print(f'Order {i}: {len(lines[mask])} lines, MAD={calc_mad(vres)} m/s')
                    thar.header_insert(f'W_NTH{i:03d}', nansafe(len(lines[mask])), f'Order {i} number of ThAr lines')
                    thar.header_insert(f'W_MAD{i:03d}', nansafe(np.round(calc_mad(vresiduals[mask]), 2)), f'Order {i} MAD of wavelength sol. (m/s)')
                # Save to ech/FITS file
                thar.save()
                
                # Save the coefficients and linelist in npz file
                # (PyReduce also already saved an npz file with a generic name, 
                # which gets overwritten in each step of this loop)
                savefile = join(savedir, thar.construct_filename(ext='.thar.npz', mode=self.mode, object=None))
                np.savez(savefile, wave=wave, coef=coef, linelist=linelist)


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

    def trace_orders(self, **kwargs):
        """Combine order trace for multi-fiber mode"""

        # Fetch single-fiber traces
        orders, column_range, order_modes = [], [], []
        for mode, calibs in self.sub_mode_calibs.items():
            # Get it if already traced, otherwise run trace on the other fiber
            sub_orders, sub_column_range = calibs.trace_orders(**kwargs)  
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
        logger.info(f"Loaded {len(orders)} order traces ({self.mode}) by combining modes {list(self.sub_mode_calibs.keys())}.")
        
        self.plot_trace(overwrite=True)

    def save_extracted(self, image, head, spec, sigma, column_range, savedir=None):
        """
        Save the extracted spectrum, ensuring that each fiber is saved to a 
        separate file and assigning the nearest wavelength solution if available
        """
        # Split extracted orders into modes/fibers
        results = []
        for sub_mode in self.sub_modes:
            selected = self.order_modes == sub_mode  # E.g. all orders from F1
            nord = np.sum(selected)
            logger.info(f'Preparing to save {nord} orders from mode {sub_mode}')

            head = header_insert(head, 'PL_OMODE', self.mode, 'Original mode (e.g. fiber)')
            sub_mode_calib = self.sub_mode_calibs[sub_mode]
            # Save this mode
            res = sub_mode_calib.save_extracted(image, head, spec[selected], 
                                                sigma[selected], column_range[selected], 
                                                savedir=savedir)
            results += res  # res is a list of one object
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

class NotEnoughFlatsError(Exception):
    pass
