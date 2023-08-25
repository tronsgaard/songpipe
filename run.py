#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import join, exists, relpath, splitext, basename, dirname
from glob import glob
import argparse
import numpy as np
import astropy.io.fits as fits

import matplotlib
matplotlib.use('Agg')  # Don't show plots..
import matplotlib.pyplot as plt

import songpipe


# Default settings 
# TODO: Move to a separate config file
defaults = {
    'basedir': '/mnt/c/data/SONG/ssmtkent/',
}

# Set up command line arguments
ap = argparse.ArgumentParser()
# Directory structure
ap.add_argument('datestr', metavar='date_string', type=str, default=None,
                help='Night date (as a string), e.g. `20220702`')
ap.add_argument('--basedir', type=str, default=defaults['basedir'],
                help=f'Base directory (default: {defaults["basedir"]})')
ap.add_argument('--rawdir', type=str, default=None,
                help=f'Override raw directory (default: <basedir>/star_spec/<date_string>/raw)')
ap.add_argument('--outdir', type=str, default=None,
                help=f'Override raw directory (default: <basedir>/extr_spec/<date_string>)')
ap.add_argument('--calibdir', type=str, default=None,
                help=f'Override calib directory (default: <basedir>/extr_spec/<date_string>/calib)')
# Actions
ap.add_argument('--plot', action='store_true',
                help='Activate plotting in PyReduce')
ap.add_argument('--reload-cache', action='store_true',
                help='Ignore cached FITS headers and reload from files')
ap.add_argument('--simple-extract', action='store_true',
                help='Extract using simple summation across orders (faster than optimal extraction)')
ap.add_argument('--silent', action='store_true',
                help='Silent mode (useful when running in background)')
# TODO: Implement these:
ap.add_argument('--prep-only', action='store_true',
                help='Prepare files only - stop before PyReduce extraction')
ap.add_argument('--ignore-existing', action='store_true',
                help='Ignore existing output files and run extraction again')

def run():
    opts = ap.parse_args()
    if opts.rawdir is None:
        # Default to <basedir>/star_spec/<date_string>/raw
        opts.rawdir = join(opts.basedir, 'star_spec', opts.datestr, 'raw')
    if opts.outdir is None:
        # Default to <basedir>/extr_spec/<date_string>
        opts.outdir = join(opts.basedir, 'extr_spec', opts.datestr)
    if opts.calibdir is None:
        # Default to <basedir>/extr_spec/<date_string>/calib
        opts.calibdir = join(opts.outdir, 'calib')

    print('SONG pipeline starting..')
    print('------------------------')
    print(f'Python version: {sys.version.split(" ")[0]}')
    print(f'songpipe version: {songpipe.__version__}')
    print(f'Raw directory:    {opts.rawdir}')
    print(f'Output directory: {opts.outdir}')
    print(f'Calib directory:  {opts.calibdir}')

    # Select image class (single channel or high/low gain)
    # TODO: This needs to be done automatically, by date or by analyzing the first FITS file
    image_class = songpipe.HighLowImage  # For Mt. Kent
    # image_class = songpipe.Image  # For Tenerife (Not fully implemented)
    print(f'Image class: <{image_class.__module__}.{image_class.__name__}>')
    print('------------------------')

    # Load all FITS headers as Image objects
    # Objects are saved to a dill file called .songpipe_cache, saving time if we need to run the pipeline again
    savename = join(opts.outdir, ".songpipe_cache")
    images = None
    if opts.reload_cache is False:
        try:
            import dill  # Similar to, yet better than, pickle
            with open(savename, 'rb') as h:
                images, version = dill.load(h)
            if version != songpipe.__version__:
                print("Cache version mismatch.")
                images = None
            else:
                print(f'Loaded FITS headers from cache: {relpath(savename, opts.outdir)}')
        except (FileNotFoundError,):
            pass
        except Exception as e:
            print(e)
            print('Could not reload FITS headers from cache')
    # If images is still None, it means we need to load the FITS headers from their source
    if images is None:
        print('Loading FITS headers from raw images...')
        # The following line loads all *.fits files from the raw directory
        images = songpipe.ImageList.from_filemask(join(opts.rawdir, '*.fits'), image_class=image_class, silent=opts.silent)
        try:
            # Save objects for next time
            import dill
            with open(savename, 'wb') as h:
                dill.dump((images, songpipe.__version__), h)
        except Exception as e:
            print(e)
            print('Could not save cache. Continuing...')

    print('------------------------')
    images.list()
    print('------------------------')
    print(f'Total: {len(images)} images')
    print('------------------------')


    # Assemble master bias
    master_bias_filename = join(opts.outdir, 'calib/master_bias.fits')
    if exists(master_bias_filename):
        filename = relpath(master_bias_filename, opts.outdir)
        print(f'Master bias already exists - loading from {filename}')
        master_bias = image_class(filename=master_bias_filename)
    else:
        print('Assembling master bias...')
        bias_list = images.filter(image_type='BIAS')
        bias_list.list()
        master_bias = bias_list.combine(method='median', silent=opts.silent)
        master_bias.save_fits(master_bias_filename, overwrite=True, dtype='float32')

    # Assemble master darks
    print('------------------------')
    print('Assembling master darks...')
    master_darks = {}  # Dict of darks for various exptimes
    exptimes = images.get_exptimes()
    print(f'Exptimes (seconds): {exptimes}')
    for exptime in exptimes:
        # For each dark exptime, construct a master dark
        master_dark_filename = join(opts.outdir, f'calib/master_dark_{exptime:.0f}s.fits')
        if exists(master_dark_filename):
            filename = relpath(master_dark_filename, opts.outdir)  # For nicer console output
            print(f'Master dark ({exptime:.0f}s) already exists - loading from {filename}')
            master_darks[exptime] = image_class(filename=master_dark_filename)
        else:
            dark_list = images.filter(image_type='DARK', exptime=exptime)  # TODO: Exptime tolerance
            if len(dark_list) == 0:
                print(f'No darks available for exptime {exptime} s')  # TODO: Handle missing darks
                continue
            print(f'Building {exptime} s master dark')
            master_darks[exptime] = dark_list.combine(method='median', silent=opts.silent)
            master_darks[exptime].subtract_bias(master_bias, inplace=True)  # Important!
            master_darks[exptime].save_fits(master_dark_filename, overwrite=True, dtype='float32')

    # Loop over all images except bias and darks
    loop_images = images.images
    loop_images = [im for im in loop_images if im.type not in ('BIAS', 'DARK')]

    # Prepare images for extraction by subtracting master bias and dark and merging high-low gain channels
    print('------------------------')
    print('Preparing images...')
    prep_images = []
    for im_orig in loop_images:
        out_filename = join(opts.outdir, 'prep', im_orig.construct_filename(suffix='prep'))
        if exists(out_filename):
            print(f'File already exists - loading from {relpath(out_filename, opts.outdir)}')
            prep_images.append(songpipe.Image(filename=out_filename))
        else:
            print(f'Reducing image: {relpath(im_orig.filename, opts.rawdir)}')
            print('Subtracting master bias...')
            im = im_orig.subtract_bias(master_bias)
            im_orig.clear_data()  # Avoid filling up the memory

            # Select master dark based on exptime
            dark_exptimes = np.array([0] + list(master_darks.keys()))
            k = np.argmin(np.abs(dark_exptimes - im.exptime))
            dark_exptime = dark_exptimes[k]  # Make this more flexible
            if dark_exptime > 0:
                print(f'Subtracting {dark_exptime}s master dark...')
                im = im.subtract_dark(master_darks[dark_exptimes[k]])

            # Apply gain
            print('Applying gain and merge high+low')
            im = im.apply_gain()  # TODO: Move gain values to instrument config
            merged = im.merge_high_low()
            im.clear_data()  # Avoid filling up the memory

            # Orientation
            print('Orienting image')
            merged = merged.orient(flip_updown=True, rotation=0)  # TODO: Move orientation parameters to instrument config

            # Save image
            merged.save_fits(out_filename, overwrite=True, dtype='float32')  # FIXME: Maybe change dtype to float64?
            merged.clear_data()  # Avoid filling up the memory

            # Append to list
            prep_images.append(merged)


        #print('----')

    # Wrap list of prepared images in ImageList class
    prep_images = songpipe.ImageList(prep_images)

    # No more need for bias and darks - clear variables to free memory..
    master_bias.clear_data()
    for k, master_dark in master_darks.items():
        master_dark.clear_data()

    print(f'Done preparing {len(prep_images)} images.')
    print('------------------------')

    ############################
    #         PyReduce         #
    ############################
    import pyreduce
    from pyreduce.configuration import get_configuration_for_instrument
    from pyreduce.instruments.common import create_custom_instrument
    from pyreduce.reduce import WavelengthCalibrationFinalize
    from pyreduce.wavelength_calibration import LineList
    from pyreduce import echelle
    from pyreduce.util import start_logging
    from pyreduce.combine_frames import combine_calibrate
    from songpipe import CalibrationSet, MultiFiberCalibrationSet  # Modified version

    print(f'Setting up PyReduce (version {pyreduce.__version__})')

    # Create custom instrument
    instrument = create_custom_instrument("SONG-Australia", mask_file=None, wavecal_file=None)
    mask = np.zeros((4096, 4096))  # TODO: Load an actual bad pixel mask
    bias, bhead = None, None  # Already subtracted

    # Load default config
    config = get_configuration_for_instrument("pyreduce", plot=opts.plot)
    
    # Modify default config
    config['wavecal']['correlate_cols'] = 512
    if opts.simple_extract:
        config['science']['collapse_function'] = 'sum' 
        config['science']['extraction_method'] = 'arc'  # SIMPLE EXTRACTION TO SPEED THINGS UP

    # Set up and link calibration modes for Mt. Kent data
    calibs = {}
    calibs['F1'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F1")
    calibs['F2'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F2")
    calibs['F12'] = MultiFiberCalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F12")
    calibs['F12'].link_single_fiber_calibs(calibs['F1'], calibs['F2'])

    # Run calibration steps via CalibrationSet objects
    for mode, calibration_set in calibs.items():
        calibration_set.combine_flats()

    for mode, calibration_set in calibs.items():
        calibration_set.trace_orders()

    # Measure scattered light from flat
    for mode, calibration_set in calibs.items():
        calibration_set.measure_scattered_light()
        calibration_set.measure_curvature()  # Dummy - not useful with fiber
        calibration_set.normalize_flat()

    # TODO: limit order range

    # Extract and calibrate all ThAr spectra
    thar_images = prep_images.filter(image_type='THAR')
    thar_images.list()

    thar_spectra = []
    for im in thar_images:
        mode = im.mode
        calibration_set = calibs[mode]
        thar_spectra += calibration_set.extract(im)

    for thar in thar_spectra:
        mode = thar.mode
        calibration_set = calibs[mode]

        head = thar.header
        data = fits.getdata(thar.filename)
        data = {column.lower(): data[column][0] for column in data.dtype.names}
        spec = data['spec']

        if 'wave' in data:
            print(f'Wavelength solution already exists for file {relpath(thar.filename, opts.outdir)}')
            wave = data['wave']        
        else:
            print(f'Wavelength calibration: {relpath(thar.filename, opts.outdir)}')

            step_args = calibration_set.step_args
            step_wavecal = WavelengthCalibrationFinalize(*step_args, **config['wavecal'])

            wavecal_master = (spec, head)
            reference = join(dirname(__file__), 'linelists/test_thar_fib2_2D.npz')
            wavecal_init = LineList.load(reference)
            wave, coef, linelist = step_wavecal.run(wavecal_master, wavecal_init)
            # Save the coefficients and linelist in npz file
            savedir = join(opts.outdir, 'wave')
            savefile = join(savedir, thar.construct_filename(ext='.thar.npz', mode=mode, object=None))
            np.savez(savefile, wave=wave, coef=coef, linelist=linelist)
            # Save .ech compatible FITS file
            data['wave'] = wave 
            echelle.save(thar.filename, head, **data)

        calibration_set.wavelength_calibs.append((head, wave))

    print('------------------------')

    # Get images to extract
    print('Finding images to extract...')
    types_to_extract = ['STAR','FLATI2','FP']
    images_to_extract = prep_images.filter(image_type=types_to_extract)
    images_to_extract.list()

    for im in images_to_extract:
        mode = im.mode
        if im.mode == 'UNKNOWN':
            mode = 'F12'  # Extract as F12 if mode is unknown (Mt. Kent)
        calibration_set = calibs[mode]
        calibration_set.extract(im, savedir=opts.outdir)

        print('------------------------')

    
    
    # Freqcomb
    # Continuum
    # Finalize

    # Output image with spectrum, blaze, wavelength (placeholder)

    print('Done!')


if __name__ == '__main__':
    run()
