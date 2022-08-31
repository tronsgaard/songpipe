#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib import reload
import sys
from os.path import join, exists
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import songpipe

datestr = '20220702'
basedir = '/mnt/c/data/SONG/ssmtkent/'
rawdir = join(basedir, 'star_spec', datestr, 'raw')
outdir = join(basedir, 'extr_spec', datestr)
image_class = songpipe.HighLowImage

print('Loading images...')
images = songpipe.ImageList.from_filemask(join(rawdir, '*fits'), image_class=image_class)
images.list()

# Assemble master bias
master_bias_filename = join(outdir, 'calib/master_bias.fits')
if exists(master_bias_filename):
    print(f'Master bias already exists - loading from {master_bias_filename}')
    master_bias = image_class(filename=master_bias_filename)
else:
    bias_list = images.filter(image_type='BIAS')
    bias_list.list()
    master_bias = bias_list.combine(method='median')
    master_bias.save_fits(master_bias_filename, overwrite=True, dtype='float32')

# Assemble master darks
master_darks = {}  # Dict of darks for various exptimes
exptimes = images.get_exptimes()
for exptime in exptimes:
    # For each dark exptime, construct a master dark
    master_dark_filename = join(outdir, f'calib/master_dark_{1e3*exptime:.0f}ms.fits')
    if exists(master_dark_filename):
        print(f'{exptime:.3f}s master dark already exists - loading from {master_dark_filename}')
        master_darks[exptime] = image_class(filename=master_dark_filename)
    else:
        print(f'Building {exptime} s master dark')
        dark_list = images.filter(image_type='DARK', exptime=exptime)  # TODO: Exptime tolerance
        if len(dark_list) == 0:
            print(f'No darks available for exptime {exptime} s')
            continue
        master_darks[exptime] = dark_list.combine(method='median')
        master_darks[exptime].subtract_bias(master_bias, inplace=True)  # Important!
        master_darks[exptime].save_fits(master_dark_filename, overwrite=True, dtype='float32')

# Loop over all images except bias and darks
loop_images = images.images
#loop_images = images.filter(exptime=240.0).images
loop_images = [im for im in loop_images if im.type not in ('BIAS', 'DARK')]
prep_images = []
for im_orig in loop_images:
    out_filename = join(outdir, 'prep', im_orig.construct_filename(suffix='prep'))
    if exists(out_filename):
        print(f'File already exists - loading from {out_filename}')
        prep_images.append(songpipe.Image(filename=out_filename))
    else:
        print(f'Reducing image: {im_orig.filename}')
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
        merged.orient(flip_updown=True, rotation=270)

        # Save image
        merged.save_fits(out_filename, overwrite=True)

        # Append to list
        prep_images.append(merged)

    print('----')