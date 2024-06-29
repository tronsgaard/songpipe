# How `songpipe` works
The intention of this document is to describe in some detail how `songpipe` works, and to act as a starting point for someone trying to figure out how to make changes or implement new instruments. 

- [How we access the core functions of `PyReduce`](#how-we-access-the-core-funcions-of-pyreduce)
- [The most important elements of `songpipe`](#the-most-important-elements-of-songpipe)
  - [How the `CalibrationSet` class works](#how-the-calibrationset-class-works)
  - [`Image` and `Spectrum`](#image-and-spectrum)
  - [Runscript structure](#runscript-structure)
- [Setting up a new instrument](#setting-up-a-new-instrument)

## How we access the core funcions of `PyReduce` 
[`PyReduce`](https://github.com/AWehrhahn/PyReduce/) ([Piskunov & Valenti (2002)](https://doi.org/10.1051/0004-6361:20020175), [Piskunov, Wehrhahn & Marquart (2020)](https://doi.org/10.1051/0004-6361/202038293)) is an excellent, versatile, and powerful echelle spectroscopy pipeline. However, the outer layers that takes care of loading and organising the data, and running the components of the pipeline, did not fulfil our needs for SONG, which led to the development of `songpipe` as a wrapper around the core functions of `PyReduce`. It is not straightforward how to do this, as `PyReduce` was designed to run on its own. The guiding philosophy has been to wrap `PyReduce` without touching or duplicating the code, although a few changes were necessary, resulting in this [fork of the code](https://github.com/tronsgaard/PyReduce/) that we use.

`PyReduce` organises the steps of the reduction process in subclasses of the `pyreduce.reduce.Step` class (e.g. `pyreduce.reduce.OrderTracing` or `pyreduce.reduce.ScienceExtraction`). Each `Step` is instantiated with the same six positional arguments: `(instrument, mode, target, night, output_dir, order_range)`, which are mainly used to set the output directory and filename. The reduction step can then be executed using the `.run()` method of the object, with the appropriate arguments.

 ```py3
# Example: Order tracing
step_args = (
                instrument,  # Instrument object
                mode,        # Mode name (e.g. F1)
                None, None,  # target name and night name, not needed
                savedir,     # Output directory
                None         # order range, not used in this step
             )
step_orders = OrderTracing(*step_args, config['orders'])
step_orders.run(flat_file, mask, bias)
 ```

In `songpipe` we handle these calls to `PyReduce` inside a class called `CalibrationSet`, which is described below. We create each `Step` object and call either their `.run()` method or the functions used within `.run()`.

## The most important elements of `songpipe`

### How the `CalibrationSet` class works
The [`songpipe.calib.CalibrationSet`](../songpipe/calib.py) class represents a collection of everything we need in order to extract a science image for a given instrument mode. Science images are extracted by feeding them to the `.extract()` method of the appropriate `CalibrationSet` instance.

When creating a `CalibrationSet` for an instrument mode, it requires a list of images that have been bias/dark corrected, cropped, and rotated to the default left-right orientation. We refer to these images as "prepared" and represent them as `Image` objects [(see below)](#image-and-spectrum). From this list, the `CalibrationSet` will pick the appropriate flats and wavelength calibrations, by inquiring each `Image` about its mode (e.g. "F1") and type (e.g. "FLAT"). We can then execute the desired calibration steps by calling methods of the `CalibrationSet`:

- `combine_flats()`
- `trace_orders()`
- `measure_scattered_light()` (not implemented yet)
- `measure_curvature()` (not implemented yet)
- `normalize_flat()`

```py3
# Example, setting up a CalibrationSet for mode F1
calibration_set = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F1")
calibration_set.combine_flats()
calibration_set.trace_orders()
calibration_set.normalize_flat()
```

ThAr wavelength calibrations are also handled inside the `CalibrationSet` object, for which we first need to add a list of extracted ThAr spectra:

```py3
# Example: Solving ThAr calibrations in mode "F1"
thar_images = prep_images.filter(image_type='THAR', mode='F1')
for im in thar_images:
   extracted_thar = calibration_set.extract(im, savedir=thardir)
   calibration_set.wavelength_calibs += extracted_thar  # This is the F1 CalibrationSet
calibration_set.solve_wavelengths(LINELIST_PATH, savedir=thardir, skip_existing=True)
```

The class `MultiFiberCalibrationSet` is a subclass of `CalibrationSet`, designed to handle the SONG-Australia setup. A `MultiFiberCalibrationSet` links to a normal `CalibrationSet` for each fiber, such that it can use the individual order trace and extract e.g. two interlaced echelle spectra (F12) into separate output files (F1 and F2).

```py3
# Example, linking modes F1 and F2 to the F12 multi-fiber calibration set
calibs = {}
calibs['F1'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F1")
calibs['F2'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F2")
calibs['F12'] = MultiFiberCalibrationSet(prep_images, calibdir, config, mask, instrument, "F12")
calibs['F12'].link_single_fiber_calibs(calibs['F1'], calibs['F2'])
```

### `Image` and `Spectrum`
[`songpipe.image.Image`](../songpipe/image.py) and [`songpipe.spectrum.Spectrum`](../songpipe/spectrum.py) objects represent images and extracted spectra, respectively. The FITS headers are cached inside the objects, and we can interrogate them via a range of properties and property methods, e.g. `.is_tenerife`, `.mode`, `.mjd_mid` etc. All properties shared between images and spectra are inherited from an abstract superclass [`songpipe.frame.Frame`](../songpipe/frame.py). These methods need to be checked and possible adapted when FITS keywords change or new instruments are implemented.

- `header`
- `object`
- `mode` : E.g. "F1", "F2", "SLIT8", "DARK", UNKNOWN", ...
- `type` : The value of IMAGETYP in header (e.g. "FLAT", "THAR", "STAR", ...)
- `exptime`
- `date_start`
- `jd_start`
- `mjd_start`
- `mjd_mid`
- `observatory`
- `is_australia` : True if `Frame` is from SONG Australia
- `is_tenerife` : True if `Frame` is from SONG Tenerife

The `Image` class contains a number of methods for calibrating images. A special class `HighLowImage` exists to accommodate the separate high- and low-gain images from the current CMOS detector at SONG-Australia.

The `Spectrum` class contains a number of property methods to access the different layers of the ECH format (`spec`, `sig`, `wave`, `cont`), the number of orders (`nord`) and the column range (`columns`).

Corresponding classes for lists of `Image` and `Spectrum` objects exist, named `ImageClass` and `SpectrumClass`. They, again, inherit from the common superclass named `FrameList`. The list classes offer useful methods for filtering lists of images or spectra, or simply printing a readable list of frames to the console or a file:

- `summarize()`
- `get_exptimes()` : List all exptime values
- `count()`
- `append()`
- `filter()`
- `list()` : Prints a pretty list of frames with definable columns
- `get_closest(mjd)` : Get the frame closest in time to `mjd`

The `ImageList` class also offers methods for combining frames.

### Runscript structure
`songpipe` runs from scripts that are adapted to each instrument, currently [`run_tenerife.py`](../run_tenerife.py) and [`run_australia.py`](../run_australia.py). The explicit nature of these scripts is intentional, as it makes it more clear what happens, how to to disable lines, or adjust parameters. It does, however, lead to some code duplication, between the runscripts.

The runscripts are powered by some shared functions from [`songpipe.running`](../songpipe/running.py), which is where console arguments, directory structure, logging etc. are defined. The code in the runscripts is nested within a function called `.run_inner()`, which is inside a function called `.run()`. This is so we can catch errors and log them before the program crashes.

## Setting up a new instrument
TBD