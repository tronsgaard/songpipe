#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import join, exists, relpath, splitext, basename
from glob import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Don't show plots..
import matplotlib.pyplot as plt
from tqdm import tqdm
import astropy.io.fits as fits
import songpipe



# Default settings (should go in a separate file)
defaults = {
    'basedir': '/mnt/c/data/SONG/ssmtkent/',
}

# Set up command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('datestr', metavar='date_string', type=str, default=None,
                help='Night date (as a string), e.g. `20220702`')
ap.add_argument('--basedir', type=str, default=defaults['basedir'],
                help=f'Base directory (default: {defaults["basedir"]})')
ap.add_argument('--rawdir', type=str, default=None,
                help=f'Override raw directory (default: <basedir>/star_spec/<date_string>/raw)')
ap.add_argument('--outdir', type=str, default=None,
                help=f'Override raw directory (default: <basedir>/extr_spec/<date_string>)')
ap.add_argument('--plot', action='store_true',
                help='Activate plotting in PyReduce')
ap.add_argument('--reload-cache', action='store_true',
                help='Ignore cached FITS headers and reload from files')
ap.add_argument('--simple-extract', action='store_true',
                help='Extract using simple summation across orders (faster than optimal extraction)')
# TODO: Implement these:
ap.add_argument('--prep-only', action='store_true',
                help='Prepare files only - stop before PyReduce extraction')
ap.add_argument('--ignore-existing', action='store_true',
                help='Ignore existing output files and run extraction again')

def run():
    opts = ap.parse_args()
    if opts.rawdir is None:
        opts.rawdir = join(opts.basedir, 'star_spec', opts.datestr, 'raw')
    if opts.outdir is None:
        opts.outdir = join(opts.basedir, 'extr_spec', opts.datestr)
    calibdir = join(opts.outdir, 'calib')

    print('SONG pipeline starting..')
    print('------------------------')
    print(f'Python version: {sys.version.split(" ")[0]}')
    print(f'Raw directory:    {opts.rawdir}')
    print(f'Output directory: {opts.outdir}')

    # Select image class (single channel or high/low gain)
    # TODO: This needs to be done automatically, by date or by analyzing the first FITS file
    image_class = songpipe.HighLowImage  # For Mt. Kent
    # image_class = songpipe.Image  # For Tenerife (Not fully implemented)
    print(f'Image class: <{image_class.__module__}.{image_class.__name__}>')
    print('------------------------')

    # Load all FITS headers as Image objects
    savename = join(opts.outdir, ".songpipe_cache")
    images = None
    if opts.reload_cache is False:
        try:
            import dill
            with open(savename, 'rb') as h:
                images = dill.load(h)
            print(f'Loaded FITS headers from cache: {relpath(savename, opts.outdir)}')
        except (FileNotFoundError,):
            pass
        except:
            print('Could not reload FITS headers from cache')
    if images is None:
        print('Loading FITS headers from raw images...')
        images = songpipe.ImageList.from_filemask(join(opts.rawdir, '*.fits'), image_class=image_class)
        try:
            # Save objects for next time
            import dill
            with open(savename, 'wb') as h:
                dill.dump(images, h)
        except:
            print('Could not save cache. Continuing...')

    print('------------------------')
    images.list()
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
        master_bias = bias_list.combine(method='median')
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
            master_darks[exptime] = dark_list.combine(method='median')
            master_darks[exptime].subtract_bias(master_bias, inplace=True)  # Important!
            master_darks[exptime].save_fits(master_dark_filename, overwrite=True, dtype='float32')

    # Loop over all images except bias and darks
    loop_images = images.images
    loop_images = [im for im in loop_images if im.type not in ('BIAS', 'DARK')]

    # Prepare images for extraction by subtracting master bias and dark and merging high-low gain channels
    print('------------------------')
    print('Preparing images (bias and dark subtraction, merging high+low gain channels)...')
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
            dark_exptime = dark_exptimes[k]
            if dark_exptime > 0:
                print(f'Subtracting {dark_exptime}s master dark...')
                im = im.subtract_dark(master_darks[dark_exptimes[k]])

            # Apply gain
            print('Applying gain and merge high+low')
            im = im.apply_gain()
            merged = im.merge_high_low()
            im.clear_data()  # Avoid filling up the memory

            # Orientation
            print('Orienting image')
            merged = merged.orient(flip_updown=True, rotation=0)

            # Save image
            merged.save_fits(out_filename, overwrite=True, dtype='float32')  # FIXME: Maybe change dtype to float64
            merged.clear_data()  # Avoid filling up the memory

            # Append to list
            prep_images.append(merged)

        #print('----')

    prep_images = songpipe.ImageList(prep_images)

    print(f'Done preparing {len(prep_images)} images.')
    print('------------------------')

    ############################
    #         PyReduce         #
    ############################
    import pyreduce
    from pyreduce.configuration import get_configuration_for_instrument
    from pyreduce.instruments.common import create_custom_instrument
    from pyreduce.reduce import Flat, OrderTracing, BackgroundScatter, NormalizeFlatField
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

    if opts.simple_extract:
        onfig['science']['collapse_function'] = 'sum' 
        config['science']['extraction_method'] = 'arc'  # SIMPLE EXTRACTION TO SPEED UP THINGS

    # Set up and link calibration modes for Mt. Kent data
    calibs = {}
    calibs['F1'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F1")
    calibs['F2'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F2")
    calibs['F12'] = MultiFiberCalibrationSet(prep_images, calibdir, config, mask, instrument, "F12")

    calibs['F12'].link_single_fiber_calibs(calibs['F1'], calibs['F2'])

    # Run calibration steps
    for mode, calibration_set in calibs.items():
        calibration_set.combine_flats()

    for mode, calibration_set in calibs.items():
        calibration_set.trace_orders()

    # Measure scattered light from flat
    for mode, calibration_set in calibs.items():
        calibration_set.measure_scattered_light()
        calibration_set.measure_curvature()  # Dummy - not useful with fiber
        calibration_set.normalize_flat()

    # TODO: set order range
        
    print('------------------------')

    # Get images to extract
    print('Finding images to extract...')
    types_to_extract = ['STAR','THAR','FLATI2','FP']
    images_to_extract = prep_images.filter(image_type=types_to_extract)
    images_to_extract.list()

    for im in images_to_extract:
        mode = im.mode
        if im.mode == 'UNKNOWN':
            mode = 'F12'  # Extract as F12 if mode is unknown (Mt. Kent)
        calibration_set = calibs[mode]
        calibration_set.extract(im, savedir=opts.outdir)


        print('------------------------')

    # Wavecal
    # Freqcomb
    # Continuum
    # Finalize

    # Output image with spectrum, blaze, wavelength (placeholder)


if __name__ == '__main__':
    run()
