#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import join, exists, relpath, dirname
from shutil import copytree
from copy import deepcopy

import songpipe.running
from songpipe.image import Image, ImageList, QHYImage
from songpipe.dark import DarkManager

# SONGpipe settings
#BASEDIR = '/mnt/brunost/song/sstenerife/'
BASEDIR = '/mnt/c/data/SONG/sstenerife/'
OBSLOG_NAME = '000_list.txt'
MIN_BIAS_IMAGES = 9  # Minimum number of bias images
MIN_DARK_IMAGES = 2   # Minimum number of dark images
MIN_DARK_EXPTIME = 10.0  # Ignore if darks are missing for exposures shorter than this (seconds)
MIN_FLAT_IMAGES = 10  # Minimum number of flat images
LINELIST_PATH = join(dirname(__file__), 'linelists/s1_qhy_1.npz')
GAIN_FACTOR = 0.096  # e-/ADU
READNOISE = 2.0  # e-

# Select image class (single channel or high/low gain)
IMAGE_CLASS = QHYImage  # Tenerife


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
    ignore_list = songpipe.running.read_ignore_list(join(opts.rawdir, '.songpipe_ignore'))
    images = songpipe.running.load_images(filemask, IMAGE_CLASS, ignore_list=ignore_list, outdir=opts.outdir, 
                                          reload_cache=opts.reload_cache, silent=opts.silent)
    
    # Ignore darks
    if opts.ignore_darks:
        images = images.filter(image_type_exclude=['BIAS', 'DARK'])
    # Ignore flats
    if opts.ignore_flats:
        images = images.filter(image_type_exclude=['FLAT'])
    # Ignore ThAr calibs
    if opts.ignore_thars:
        images = images.filter(image_type_exclude=['THAR'])

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
    dark_manager = DarkManager([], image_class=Image, 
                               combine_method='median', savedir=opts.darkdir, 
                               min_dark_images=MIN_DARK_IMAGES, 
                               min_bias_images=MIN_BIAS_IMAGES)
    
    # Add fallback master darks (and master biases) obtained on previous nights
    for d in opts.add_darks:
        d = join(d, '*.fits')
        logger.info(f'Loading additional master darks from {d}')
        dark_manager.append_from_filemask(d)

    # Now build master bias and master darks from all available exposure times
    if opts.ignore_darks:
        logger.info('Ignoring darks and bias frames from this night')
    else:
        dark_manager.build_master_bias(images, silent=opts.silent)
        #dark_manager.build_all_master_darks(images, silent=opts.silent)

    # Check if we have all the needed master darks
    #dark_manager.check_exptimes(images.get_exptimes(), min_exptime=MIN_DARK_EXPTIME)

    # Now we can request the master bias and master dark like this:
    master_bias = dark_manager.get_master_bias()
    # master_dark = dark_manager.get_master_dark(60)

    # PREPARE IMAGES
    # Prepare for extraction by subtracting master bias and dark
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
            logger.info(f'File already exists - loading from {relpath(out_filename, opts.outdir)}')
            im = Image(filename=out_filename)
            prep_images.append(im)
        else:
            logger.info(f'Reducing image: {relpath(im_orig.filename, opts.rawdir)}')
            logger.info('Subtracting master bias...')
            im = im_orig.subtract_bias(master_bias)
            im_orig.clear_data()  # Avoid filling up the memory

        #    # Get master dark for exptime
        #    if im.exptime > MIN_DARK_EXPTIME:
        #        master_dark = dark_manager.get_master_dark(im.exptime, im.mjd_mid)
        #        
        #        # Subtract master dark
        #        logger.info(f'Subtracting master dark: {master_dark.filename}')
        #        im = im.subtract_dark(master_dark)

            # Apply gain
            #logger.info('Applying gain')
            #gain_factor = im.get_header_value('GAIN')
            #im = im.apply_gain(GAIN_FACTOR) 

            # Orientation
            logger.info('Orienting image')
            im = im.orient(flip_leftright=True)  # No rotation

            # Save image
            im.save_fits(out_filename, overwrite=True, dtype='float32')  # FIXME: Maybe change dtype to float64?
            im.clear_data()  # Avoid filling up the memory

            # Append to list
            prep_images.append(im)


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
    #from pyreduce.instruments.common import create_custom_instrument
    from songpipe.calib import CalibrationSet, NotEnoughFlatsError
    from songpipe.spectrum import SpectrumList
    from songpipe.instruments import SONGInstrument

    logger.manager.loggerDict['pyreduce'].handlers.clear()  # Pyreduce sets its own logging handler, resulting in duplicate output if we don't clear it
    logger.info(f'Setting up PyReduce (version {pyreduce.__version__})')

    # Create custom instrument
    #instrument = create_custom_instrument("SONG-Tenerife", mask_file=None, wavecal_file=None)
    instrument = SONGInstrument('SONG-Tenerife', modes=['SLIT2', 'SLIT5', 'SLIT6', 'SLIT8'])
    instrument.info['gain'] = GAIN_FACTOR
    instrument.info['readno'] = READNOISE
    instrument.info['dark'] = 0
    
    mask = np.zeros((master_bias.shape))  # TODO: Load an actual bad pixel mask

    # Load default config
    config = get_configuration_for_instrument("pyreduce", plot=opts.plot)
    
    # Modify default config
    config['orders']['min_cluster'] = 100000  # Minimum number of pixels in each cluster
    config['orders']['border_width'] = 0  # excluded rows top and bottom
    config['orders']['merge_min_threshold'] = 0.9  # don't merge any orders
    
    #config['norm_flat']['threshold'] = 3200 # Temporary - debugging!
    config['norm_flat']['smooth_slitfunction'] = 100       # 2
    config['norm_flat']['smooth_spectrum']     = 1e-7       # 2
    config['norm_flat']['extraction_width']    = 0.5 
    config['norm_flat']['oversampling']    = 5 
    config['norm_flat']['maxiter']         = 40 
    config['norm_flat']['swath_width']     = 600
    config['norm_flat']['plot']            = False

    config['science']['extraction_width'] = 0.4
    config['science']['oversampling']     = 1
    config['science']['smooth_slitfunction']= 0.1
    config['science']['smooth_spectrum'] = 0

    config['wavecal_master']['extraction_width'] = 0.4
    config['wavecal_master']['collapse_function'] = 'sum'
    config['wavecal']['correlate_cols'] = 512
    config['wavecal']['threshold']      = 2500
    config['wavecal']['degree']         = [ 5, 6 ]
    config['wavecal']['iterations']     = 5
    config['wavecal']['medium']         = 'air'

    config_pinhole = deepcopy(config)
    config_pinhole['orders']['min_cluster'] = 4000  # Minimum number of pixels in each cluster
    
    if opts.simple_extract:
        config['science']['collapse_function'] = 'sum' 
        config['science']['extraction_method'] = 'arc'

    # Dump config to json file
    import json
    json_outfile = join(opts.logdir, 'config.json')
    with open(json_outfile, 'w') as h:
        h.write(json.dumps(config, indent=2))

    # Copy flat calibrations (trace, normflat, scatter) from another night
    if opts.copy_calibs is not None:
        src = opts.copy_calibs
        dest = opts.calibdir
        logger.info(f'Copying existing calibrations from {src} to {dest}')
        copytree(src, dest, dirs_exist_ok=True)

    # Set up and link calibration modes for Tenerife
    calibs = {}
    calibs['SLIT8'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "SLIT8", mode_in_extracted_filenames=False)
    calibs['SLIT6'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "SLIT6", mode_in_extracted_filenames=False)
    calibs['SLIT5'] = CalibrationSet(prep_images, opts.calibdir, config, mask, instrument, "SLIT5", mode_in_extracted_filenames=False)
    calibs['SLIT2'] = CalibrationSet(prep_images, opts.calibdir, config_pinhole, mask, instrument, "SLIT2", mode_in_extracted_filenames=False)

    # Run calibration steps via CalibrationSet objects
    for mode in list(calibs.keys()):  # We loop on a copy of calibs.keys(), as we may need to remove elements from calibs while looping
        try:
            calibration_set = calibs[mode]
            calibration_set.combine_flats(min_flat_images=MIN_FLAT_IMAGES)
        except NotEnoughFlatsError as e:
            logger.exception(e)
            logger.warning(f'Not enough flats in mode "{mode}". Continuing without this mode.')
            del calibs[mode]

    for mode, calibration_set in calibs.items():
        calibration_set.trace_orders(ymin=120., ymax=None, target_nord=46)
        calibration_set.log_extraction_widths()

    # Measure scattered light from flat
    for mode, calibration_set in calibs.items():
        #calibration_set.measure_scattered_light()
        calibration_set.data['scatter'] = None  # FIXME: Implement scattered light step
        calibration_set.measure_curvature()  # FIXME: Implement curvature step
        calibration_set.normalize_flat()

    # Extract and calibrate all ThAr spectra
    logger.info('Preparing to extract and calibrate ThAr spectra')
    thar_images = prep_images.filter(image_type='THAR')
    thar_images.list()

    thar_spectra = SpectrumList([])
    for mode, calibration_set in calibs.items():
        # For each mode/calibration set, extract all ThAr spectra
        for im in thar_images.filter(mode=mode):
            result = calibration_set.extract(im, savedir=opts.thardir)
            thar_spectra += result  # extract() always outputs a list of spectra

    # Loop over modes again, such that F12 spectra get solved individually as F1 and F2
    for mode, calibration_set in calibs.items():
        # Store list of extracted ThAr spectra in calibration set
        calibration_set.wavelength_calibs += thar_spectra.filter(mode=mode)
        # Solve wavelengths for each extracted spectrum
        calibration_set.solve_wavelengths(LINELIST_PATH, savedir=opts.thardir, skip_existing=True)

    # Add fallback ThAr calibs from different nights
    for d in opts.add_thars:
        d = join(d, '*.fits')
        logger.info(f'Loading additional ThAr calibs from {d}')
        additional_thars = SpectrumList.from_filemask(d)
        additional_thars = additional_thars.filter(image_type='THAR')
        additional_thars.list()
        for mode, calibration_set in calibs.items():
            calibration_set.wavelength_calibs += additional_thars.filter(mode=mode)

    # Extract summed flats
    for mode, calibration_set in calibs.items():
        calibration_set.extract_flat()

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
        try:
            calibration_set = calibs[im.mode]
        except KeyError:
            logger.warning(f'No calibrations for mode {im.mode}')
            continue
        calibration_set.extract(im, savedir=opts.stardir)

    if opts.extract is None:
        # Extract FlatI2
        if opts.skip_flati2 is not True:
            for im in prep_images.filter(image_type='FLATI2'):
                try:
                    calibration_set = calibs[im.mode]
                except KeyError:
                    logger.warning(f'No calibrations for mode {im.mode}')
                    continue
                calibration_set.extract(im, savedir=opts.flati2dir)

        # Extract FP
        if opts.skip_fp is not True:
            for im in prep_images.filter(image_type='FP'):
                try:
                    calibration_set = calibs[im.mode]
                except KeyError:
                    logger.warning(f'No calibrations for mode {im.mode}')
                    continue
                calibration_set.extract(im, savedir=opts.fpdir)

    print('------------------------')

    # Freqcomb
    # Continuum
    # Finalize

    logger.info('Done!')


if __name__ == '__main__':
    run()
