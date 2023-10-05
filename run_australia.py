#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join, exists, relpath, dirname
from os import makedirs

import songpipe.running
from songpipe.image import Image, HighLowImage, ImageList

# SONGpipe settings
BASEDIR = '/mnt/c/data/SONG/ssmtkent/'
MIN_BIAS_IMAGES = 11  # Minimum number of bias images
MIN_DARK_IMAGES = 5   # Minimum number of dark images
MIN_DARK_EXPTIME = 10.0  # Ignore if darks are missing for exposures shorter than this (seconds)
MIN_FLAT_IMAGES = 11  # Minimum number of flat images

# Select image class (single channel or high/low gain)
# TODO: Should this be done automatically, by date or by analyzing the first FITS file?
IMAGE_CLASS = HighLowImage  # For Mt. Kent
# IMAGE_CLASS = Image  # For Tenerife (Not fully implemented)


def run():
    """This function is called when running this file from command line"""

    # This function defines and parses the command line arguments
    opts = songpipe.running.parse_arguments(BASEDIR)

    # Set up logging
    log_file = join(opts.logdir, 'songpipe.log')
    logger = songpipe.running.setup_logger(log_file, silent=opts.silent, debug=opts.debug)

    # Log info about parameters and software versions
    songpipe.running.log_summary(opts, IMAGE_CLASS)

    # Catch and log any error arising within the run_inner() function
    try:
        run_inner(opts, logger)
    except Exception as e:
        logger.exception(e)

def run_inner(opts, logger):
    """The inner run function ensures that we can catch errors and log them properly, even when using silent mode"""
    # Heavier import moved here, to make the program start faster
    import numpy as np
    import astropy.io.fits as fits

    import matplotlib
    matplotlib.use('Agg')  # Don't show plots..
    import matplotlib.pyplot as plt

    # This function loads the header of every FITS file matching the filemask 
    # and returns a list of <Image> objects. Additionally, the objects are stored
    # to disk in a single file (<outdir>/.songpipe_cache) using `dill` (similar to `pickle`).
    # This speeds up the loading if the pipeline needs to run again.
    filemask = join(opts.rawdir, '*.fits')
    images = songpipe.running.load_images(filemask, IMAGE_CLASS, outdir=opts.outdir, 
                                          reload_cache=opts.reload_cache, silent=opts.silent)

    # MASTER BIAS
    # Define where the master bias should be saved/loaded from
    master_bias_filename = join(opts.outdir, 'prep/master_bias.fits')
    if exists(master_bias_filename):
        # Filename relative to outdir (for logging only)
        filename = relpath(master_bias_filename, opts.outdir)
        # Load master bias with proper image class
        logger.info(f'Master bias already exists - loading from {filename}')
        master_bias = IMAGE_CLASS(filename=master_bias_filename)
    else:
        logger.info('Assembling master bias...')
        # Fetch a list of bias images
        bias_list = images.filter(image_type='BIAS')
        bias_list.list()
        # Assert that we have enough images
        if len(bias_list) < MIN_BIAS_IMAGES:
            raise Exception(f'Not enough bias images. Expected {MIN_BIAS_IMAGES}, found {len(bias_list)}')
        # Assemble master bias using specified combine function
        master_bias = bias_list.combine(method='median', silent=opts.silent)
        master_bias.save_fits(master_bias_filename, overwrite=True, dtype='float32')

    # MASTER DARKS
    logger.info('------------------------')
    # Get list of all exptimes
    exptimes = images.get_exptimes()
    logger.info(f'Exptimes (seconds): {exptimes}')

    master_darks = {}  # Dict of master dark for each exptime
    for exptime in exptimes:
        # Define where the master dark for this exptime should be saved/loaded from
        master_dark_filename = join(opts.outdir, f'prep/master_dark_{exptime:.0f}s.fits')
        if exists(master_dark_filename):
            # Filename relative to outdir (for logging only)
            filename = relpath(master_dark_filename, opts.outdir)  # For nicer console output
            logger.info(f'Master dark ({exptime:.0f}s) already exists - loading from {filename}')
            # Load master dark with proper image class
            master_darks[exptime] = IMAGE_CLASS(filename=master_dark_filename)
        else:
            logger.info(f'Assembling {exptime} s master dark')
            dark_list = images.filter(image_type='DARK', exptime=exptime)  # TODO: Exptime tolerance
            # Assert that we have enough images
            if len(dark_list) < MIN_DARK_IMAGES and exptime >= MIN_DARK_EXPTIME:
                raise Exception(f'Not enough dark images for {exptime} s master dark. Expected {MIN_DARK_IMAGES}, found {len(dark_list)}')
            # Handle case of zero darks if 
            if len(dark_list) == 0:
                logger.warning(f'No darks available for exptime {exptime} s')  # TODO: Handle missing darks
                continue

            # Assemble master dark using the specified combine function
            master_darks[exptime] = dark_list.combine(method='median', silent=opts.silent)
            master_darks[exptime].subtract_bias(master_bias, inplace=True)  # Important!
            master_darks[exptime].save_fits(master_dark_filename, overwrite=True, dtype='float32')

    # PREPARE IMAGES
    # Prepare for extraction by subtracting master bias and dark, then merge high-low gain channels
    logger.info('------------------------')
    logger.info('Preparing images...')
    prep_images = []
    # Loop over all images except bias and darks
    loop_images = images.filter(image_type_exclude=('BIAS', 'DARK'))
    for im_orig in loop_images.images:
        # Define where this prepared image should be saved/loaded from
        out_filename = join(opts.outdir, 'prep', im_orig.construct_filename(suffix='prep'))
        if exists(out_filename):
            logger.debug(f'File already exists - loading from {relpath(out_filename, opts.outdir)}')
            im = Image(filename=out_filename)
            prep_images.append(im)
        else:
            logger.info(f'Reducing image: {relpath(im_orig.filename, opts.rawdir)}')
            logger.info('Subtracting master bias...')
            im = im_orig.subtract_bias(master_bias)
            im_orig.clear_data()  # Avoid filling up the memory

            # Select master dark based on exptime
            dark_exptimes = np.array([0] + list(master_darks.keys()))
            k = np.argmin(np.abs(dark_exptimes - im.exptime))
            dark_exptime = dark_exptimes[k]  # TODO: Make this more flexible
            if dark_exptime > 0:
                logger.info(f'Subtracting {dark_exptime}s master dark...')
                im = im.subtract_dark(master_darks[dark_exptimes[k]])

            # Apply gain
            logger.info('Applying gain and merge high+low')
            im = im.apply_gain()  # TODO: Move gain values to instrument config
            merged = im.merge_high_low()
            im.clear_data()  # Avoid filling up the memory

            # Orientation
            logger.info('Orienting image')
            merged = merged.orient(flip_updown=True, rotation=0)  # TODO: Move orientation parameters to instrument config

            # Save image
            merged.save_fits(out_filename, overwrite=True, dtype='float32')  # FIXME: Maybe change dtype to float64?
            merged.clear_data()  # Avoid filling up the memory

            # Append to list
            prep_images.append(merged)


        #print('----')

    # Wrap list of prepared images in ImageList class
    prep_images = ImageList(prep_images)

    # No more need for bias and darks - clear variables to free memory..
    master_bias.clear_data()
    for k, master_dark in master_darks.items():
        master_dark.clear_data()

    logger.info(f'Done preparing {len(prep_images)} images.')
    logger.info('------------------------')

    ############################
    #         PyReduce         #
    ############################
    import pyreduce
    from pyreduce.configuration import get_configuration_for_instrument
    from pyreduce.instruments.common import create_custom_instrument
    from pyreduce.reduce import WavelengthCalibrationFinalize
    from pyreduce.wavelength_calibration import LineList
    from pyreduce import echelle
    from songpipe.calib import CalibrationSet, MultiFiberCalibrationSet  # Modified version

    log_file = join(opts.logdir, 'songpipe.log')
    logger = songpipe.running.setup_logger(log_file, silent=opts.silent)  # Do this again to remove the pyreduce logger that loads on import
    logger.info(f'Setting up PyReduce (version {pyreduce.__version__})')

    # Create custom instrument
    instrument = create_custom_instrument("SONG-Australia", mask_file=None, wavecal_file=None)
    mask = np.zeros((4096, 4096))  # TODO: Load an actual bad pixel mask

    # Load default config
    config = get_configuration_for_instrument("pyreduce", plot=opts.plot)
    
    # Modify default config
    config['wavecal']['correlate_cols'] = 512
    config['wavecal']['threshold'] = 1000
    config['norm_flat']['smooth_slitfunction'] = 2 
    if opts.simple_extract:
        config['science']['collapse_function'] = 'sum' 
        config['science']['extraction_method'] = 'arc'  # SIMPLE EXTRACTION TO SPEED THINGS UP

    # Dump config to json file
    import json
    json_outfile = join(opts.logdir, 'config.json')
    with open(json_outfile, 'w') as h:
        h.write(json.dumps(config, indent=2))

    # Set up and link calibration modes for Mt. Kent data
    calibs = {}
    calibs['F1'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F1")
    calibs['F2'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F2")
    calibs['F12'] = MultiFiberCalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "F12")
    calibs['F12'].link_single_fiber_calibs(calibs['F1'], calibs['F2'])

    # Run calibration steps via CalibrationSet objects
    for mode, calibration_set in calibs.items():
        calibration_set.combine_flats(min_flat_images=MIN_FLAT_IMAGES)

    for mode, calibration_set in calibs.items():
        #ymin, ymax = (156., 3766.)  # 69 orders from 4211 - 7971 Å 
        calibration_set.trace_orders(ymin=156., ymax=3766., target_nord=69)

    # Measure scattered light from flat
    for mode, calibration_set in calibs.items():
        calibration_set.measure_scattered_light()
        calibration_set.measure_curvature()  # Dummy - not useful with fiber
        calibration_set.normalize_flat()

    # Extract and calibrate all ThAr spectra
    thar_images = prep_images.filter(image_type='THAR')
    thar_images.list()

    thardir = join(opts.outdir, 'thar')
    makedirs(thardir, exist_ok=True)

    thar_spectra = []
    for im in thar_images:
        mode = im.mode
        calibration_set = calibs[mode]
        thar_spectra += calibration_set.extract(im, savedir=thardir)

    for thar in thar_spectra:
        mode = thar.mode
        calibration_set = calibs[mode]

        head = thar.header
        data = fits.getdata(thar.filename)
        data = {column.lower(): data[column][0] for column in data.dtype.names}
        spec = data['spec']

        if 'wave' in data:
            logger.info(f'Wavelength solution already exists for file {relpath(thar.filename, opts.outdir)}')
            wave = data['wave']
        else:
            logger.info(f'Wavelength calibration: {relpath(thar.filename, opts.outdir)}')

            #step_args = calibration_set.step_args
            # (instrument, mode, target, night, output_dir, order_range)
            step_args = (instrument, mode, None, None, thardir, calibration_set.order_range)
            step_wavecal = WavelengthCalibrationFinalize(*step_args, **config['wavecal'])

            wavecal_master = (spec, head)
            reference = join(dirname(__file__), 'linelists/test_thar_fib2_2D.npz')
            wavecal_init = LineList.load(reference)
            wave, coef, linelist = step_wavecal.run(wavecal_master, wavecal_init)
            # Save the coefficients and linelist in npz file
            savefile = join(thardir, thar.construct_filename(ext='.thar.npz', mode=mode, object=None))
            np.savez(savefile, wave=wave, coef=coef, linelist=linelist)
            # Save .ech compatible FITS file
            data['wave'] = wave 
            echelle.save(thar.filename, head, **data)

        calibration_set.wavelength_calibs.append((head, wave))

    # Extract specific file
    if opts.extract is not None:
        files_to_extract = ImageList.from_filemask(opts.extract)
    else:
        # Extract star spectra
        files_to_extract = prep_images.filter(image_type='STAR')

    for im in files_to_extract:
        mode = im.mode
        if im.mode == 'UNKNOWN':
            mode = 'F12'  # Extract as F12 if mode is unknown (Mt. Kent)
        calibration_set = calibs[mode]
        calibration_set.extract(im, savedir=join(opts.outdir, 'star'))

    if opts.extract is None:
        # Extract FlatI2
        if opts.skip_flati2 is not True:
            for im in prep_images.filter(image_type='FLATI2'):
                calibration_set = calibs[im.mode]
                calibration_set.extract(im, savedir=join(opts.outdir, 'flati2'))

        # Extract FP
        if opts.skip_fp is not True:
            for im in prep_images.filter(image_type='FP'):
                calibration_set = calibs[im.mode]
                calibration_set.extract(im, savedir=join(opts.outdir, 'fp'))

    print('------------------------')

    
    
    # Freqcomb
    # Continuum
    # Finalize

    # Output image with spectrum, blaze, wavelength (placeholder)

    logger.info('Done!')


if __name__ == '__main__':
    run()