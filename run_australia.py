#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import join, exists, relpath, dirname
from os import makedirs

import songpipe.running
from songpipe.image import Image, HighLowImage, ImageList
from songpipe.dark import DarkManager

# SONGpipe settings
BASEDIR = '/mnt/c/data/SONG/ssmtkent/'
OBSLOG_NAME = '000_list.txt'
MIN_BIAS_IMAGES = 11  # Minimum number of bias images
MIN_DARK_IMAGES = 5   # Minimum number of dark images
MIN_DARK_EXPTIME = 10.0  # Ignore if darks are missing for exposures shorter than this (seconds)
MIN_FLAT_IMAGES = 11  # Minimum number of flat images
LINELIST_PATH = join(dirname(__file__), 'linelists/test_thar_fib2_2D.npz')

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
    
    # Print and store list of observations ("obslog")
    if opts.skip_obslog is False:
        # Write to file and console simultaneously
        images.list(outfile=join(opts.outdir, OBSLOG_NAME), silent=opts.silent)
    elif opts.silent is False:
        # Only print in console
        images.list()
    
    if opts.obslog_only is True:
        logger.info('Done! (obslog only)')
        sys.exit()

    # MASTER BIAS AND DARKS
    # Initialize the DarkManager, from which we will request the darks we need
    dark_manager = DarkManager([], image_class=IMAGE_CLASS, 
                               combine_method='median', savedir=opts.darkdir, 
                               min_dark_images=MIN_DARK_IMAGES, 
                               min_bias_images=MIN_BIAS_IMAGES)
    
    # Add fallback master darks (and master biases) obtained on previous nights
    for d in opts.add_darks:
        d = join(d, '*.fits')
        logger.info(f'Loading additional master darks from {d}')
        dark_manager.append_from_filemask(d)

    # Now build master bias and master darks from all available exposure times
    dark_manager.build_master_bias(images, silent=opts.silent)
    dark_manager.build_all_master_darks(images, silent=opts.silent)

    # Check if we have all the needed master darks
    dark_manager.check_exptimes(images.get_exptimes(), min_exptime=MIN_DARK_EXPTIME)

    # Now we can request the master bias and master dark like this:
    master_bias = dark_manager.get_master_bias()
    # master_dark = dark_manager.get_master_dark(60)

    # PREPARE IMAGES
    # Prepare for extraction by subtracting master bias and dark, then merge high-low gain channels
    logger.info('------------------------')
    logger.info('Preparing images...')
    prep_images = []
    # Loop over all images except bias and darks (and except FLATI2 and FP if set)
    loop_images = images.filter(image_type_exclude=('BIAS', 'DARK'))
    if opts.skip_flati2 is True:
        loop_images = images.filter(image_type_exclude='FLATI2')
    if opts.skip_fp is True:
        loop_images = images.filter(image_type_exclude='FP')

    for im_orig in loop_images.images:
        # Define where this prepared image should be saved/loaded from
        out_filename = join(opts.prepdir, im_orig.construct_filename(suffix='prep'))
        if exists(out_filename):
            logger.debug(f'File already exists - loading from {relpath(out_filename, opts.outdir)}')
            im = Image(filename=out_filename)
            prep_images.append(im)
        else:
            logger.info(f'Reducing image: {relpath(im_orig.filename, opts.rawdir)}')
            logger.info('Subtracting master bias...')
            im = im_orig.subtract_bias(master_bias)
            im_orig.clear_data()  # Avoid filling up the memory

            # Get master dark for exptime
            if im.exptime > MIN_DARK_EXPTIME:
                master_dark = dark_manager.get_master_dark(im.exptime, im.mjd_mid)
                
                # Subtract master dark
                logger.info(f'Subtracting master dark: {master_dark.filename}')
                im = im.subtract_dark(master_dark)

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
    dark_manager.clear_data()

    logger.info(f'Done preparing {len(prep_images)} images.')
    logger.info('------------------------')

    ############################
    #         PyReduce         #
    ############################
    import pyreduce
    from pyreduce.configuration import get_configuration_for_instrument
    from pyreduce.instruments.common import create_custom_instrument
    from pyreduce.wavelength_calibration import LineList
    from songpipe.calib import CalibrationSet, MultiFiberCalibrationSet  # Modified version
    from songpipe.spectrum import SpectrumList

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
    config['science']['extraction_width'] = 0.4
    config['wavecal_master']['extraction_width'] = 0.4
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
        # TODO: Move below settings somewhere else
        #ymin, ymax = (156., 3766.)  # 69 orders from 4211 - 7971 Å 
        calibration_set.trace_orders(ymin=156., ymax=3766., target_nord=69)
        calibration_set.log_extraction_widths()

    # Measure scattered light from flat
    for mode, calibration_set in calibs.items():
        calibration_set.measure_scattered_light()
        calibration_set.measure_curvature()  # Dummy - not useful with fiber
        calibration_set.normalize_flat()

    # Extract and calibrate all ThAr spectra
    logger.info('Preparing to extract and calibrate ThAr spectra')
    thar_images = prep_images.filter(image_type='THAR')
    thar_images.list()

    thar_spectra = []
    for mode, calibration_set in calibs.items():
        # For each mode/calibration set, extract all ThAr spectra
        for im in thar_images.filter(mode=mode):
            # FIXME: Use the right config for ThAr extraction?
            result = calibration_set.extract(im, savedir=opts.thardir)
            thar_spectra += result  # extract() always outputs a list of spectra
    
    # Turn the list into a SpectrumList object
    thar_spectra = SpectrumList(thar_spectra)

    # Loop over modes again, such that F12 spectra get solved individually as F1 and F2
    for mode, calibration_set in calibs.items():
        # Store list of extracted ThAr spectra in calibration set
        calibration_set.thar_spectra = thar_spectra.filter(mode=mode)
        linelist = LineList.load(LINELIST_PATH)
        # Solve wavelengths for each extracted spectrum
        calibration_set.solve_wavelengths(linelist, savedir=opts.thardir, skip_existing=True)

    # Exit if flag --calib-only is set
    if opts.calib_only is True:
        sys.exit()

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
        calibration_set.extract(im, savedir=opts.stardir)

    if opts.extract is None:
        # Extract FlatI2
        if opts.skip_flati2 is not True:
            for im in prep_images.filter(image_type='FLATI2'):
                calibration_set = calibs[im.mode]
                calibration_set.extract(im, savedir=opts.flati2dir)

        # Extract FP
        if opts.skip_fp is not True:
            for im in prep_images.filter(image_type='FP'):
                calibration_set = calibs[im.mode]
                calibration_set.extract(im, savedir=opts.fpdir)

    print('------------------------')

    
    
    # Freqcomb
    # Continuum
    # Finalize

    # Output image with spectrum, blaze, wavelength (placeholder)

    logger.info('Done!')


if __name__ == '__main__':
    run()
