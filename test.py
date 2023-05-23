#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import join, exists, relpath, splitext, basename
from glob import glob
import argparse
import numpy as np
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

    print('Loading FITS headers from raw images...')
    images = songpipe.ImageList.from_filemask(join(opts.rawdir, '*.fits'), image_class=image_class)
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

            # Orientation
            print('Orienting image')
            merged = merged.orient(flip_updown=True, rotation=0)

            # Save image
            merged.save_fits(out_filename, overwrite=True, dtype='float32')  # FIXME: Maybe change dtype to float64

            # Clear cached data
            merged.clear_data()
            im.clear_data()

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
    from songpipe import CustomScienceExtraction  # Modified version

    print(f'Setting up PyReduce (version {pyreduce.__version__})')

    # Create custom instrument
    instrument = create_custom_instrument("SONG-Australia", mask_file=None, wavecal_file=None)
    mask = np.zeros((4096, 4096))  # TODO: Load an actual bad pixel mask

    # Load default config
    config = get_configuration_for_instrument("pyreduce", plot=1)

    order_range = None  # Process all orders

    # Get flat exposures for each mode (F1, F2, F12)
    modes = ['F1', 'F2', 'F12']
    flats = {}
    flats['F1'] = prep_images.filter(image_type='FLAT', object_exact='FLATfib1')
    flats['F2'] = prep_images.filter(image_type='FLAT', object_exact='FLATfib2')
    flats['F12'] = prep_images.filter(image_type=('FLAT'), object_exact='FLATfib12')

    # Calibration loop (loop over each mode)
    for mode in modes:

        # Combine the flats
        step_flat = Flat(instrument, mode, None, opts.datestr, calibdir, None, **config['flat'])
        if exists(step_flat.savefile):
            flat, flat_header = step_flat.load(mask)
            print(f"Loaded existing master flat ({mode}) from {relpath(step_flat.savefile, opts.outdir)}.")
        else:
            print(f'Assembling master flat ({mode})...')
            flat, flat_header = step_flat.run(flats[mode].files, None, mask)

        # Order tracing
        step_orders = OrderTracing(instrument, mode, None, opts.datestr, calibdir, order_range, **config['orders'])
        if exists(step_orders.savefile):
            orders, column_range = step_orders.load()
            print(f"Loaded {len(orders)} order traces ({mode}) from {relpath(step_orders.savefile, opts.outdir)}.")
        else:
            print(f'Tracing orders in ({mode})...')
            orders, column_range = step_orders.run([step_flat.savefile], mask, None)
            print(f"Traced {len(orders)} orders in image {relpath(step_flat.savefile, opts.outdir)}.")

        # Custom order plot
        name, _ = splitext(step_orders.savefile)
        savename = name + '.png'
        songpipe.plot_order_trace(flat, orders, column_range, savename=savename)  # Custom plot routine
        print(f'Order plot saved to: {savename}')

        # Curvature
        # -- not necessary with fiber mode
        curvature = (None, None)

        # Measure scattered light from flat
        print(f'Measuring scattered light in master flat...')
        step_scatter = BackgroundScatter(instrument, mode, None, opts.datestr, calibdir, order_range,
                                         **config['scatter'])
        if exists(step_scatter.savefile):
            scatter = step_scatter.load()
            print('Loaded existing scattered light fit for image {}')
        else:
            print('Measuring scattered light')
            scatter = step_scatter.run([step_flat.savefile], mask, None, (orders, column_range))

        # TODO: Re-trace orders?

        # Norm flat and blaze
        step_normflat = NormalizeFlatField(instrument, mode, None, opts.datestr, calibdir, order_range, **config['norm_flat'])
        try:
            norm, blaze = step_normflat.load()
            print(f'Loaded existing normflat ({mode}) from {relpath(step_normflat.savefile, opts.outdir)}')
        except FileNotFoundError:
            print(f'Normalizing {mode} flat field...')
            norm, blaze = step_normflat.run((flat, flat_header), (orders, column_range), scatter, curvature)

        print('------------------------')

        # Get images to extract in this mode
        print('Finding images to extract...')
        types_to_extract = ['STAR','THAR','FLATI2','FP']
        modes_to_extract = [mode]
        if mode == 'F12':
            modes_to_extract += ['UNKNOWN']  # Extract unknown mode as F12
        images_to_extract = prep_images.filter(mode=modes_to_extract, image_type=types_to_extract)
        images_to_extract.list()
        print('------------------------')
        if len(images_to_extract) == 0:
            print('No images to extract in mode {mode}.')
            print('------------------------')
            continue

        print(f'Extracting {len(images_to_extract)} frames in mode {mode}...')
        target = None

        step_science = CustomScienceExtraction(instrument, mode, target, opts.datestr, opts.outdir, order_range,
                                               **config['science'])
        for f in images_to_extract:
            print(f'Working on file: {basename(f.filename)}')
            outfile = step_science.science_file(f.filename)
            if exists(outfile):
                print(f'Extracted spectrum already exists: {relpath(outfile, opts.outdir)}')
                print('Skipping...')
            else:
                heads, specs, sigmas, columns = step_science.run([f.filename], None, (orders, column_range),
                                                                 (norm, blaze), curvature, mask)
                # TODO: Generate diagnostic plot
        print('------------------------')

    # Wavecal
    # Freqcomb
    # Continuum
    # Finalize

    # Output image with spectrum, blaze, wavelength (placeholder)


if __name__ == '__main__':
    run()
