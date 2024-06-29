# How `songpipe` works
The intention of this document is to describe in some detail how `songpipe` works, and to act as a starting point for someone trying to figure out how to make changes or implement new instruments. 

## How we access the core funcions of `PyReduce` 
 `PyReduce` is an excellent, versatile, and powerful echelle spectroscopy pipeline. The substantial development efforts that went into developing `PyReduce` and its parent, the IDL REDUCE pipeline, are highly appreciated. However, the outer layer that takes care of loading and organising the data, and running the components of the pipeline, did not fulfil our needs for SONG, which led to the development of `songpipe` as a wrapper around the core functions of `PyReduce`. It is not straightforward how to do this, as `PyReduce` was designed to run on its own. The overarching idea has been to wrap `PyReduce` without touching the code, although a few changes were necessary, resulting in this [fork of the code](https://github.com/tronsgaard/PyReduce/) that we use.

 `PyReduce` organises the steps if the reduction process in subclasses of the `pyreduce.reduce.Step` class, e.g. `pyreduce.reduce.OrderTracing` or `pyreduce.reduce.ScienceExtraction`. Each `Step` is instantiated with the same six positional arguments: `(instrument, mode, target, night, output_dir, order_range)`, which are mainly used to set the output directory and filename. Then the reduction step can be executed using the `.run()` method, with the appropriate arguments.

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

In `songpipe` we handle the calls to `PyReduce` inside a class called `CalibrationSet`, which is described below. We create each `Step` object and call either their `.run()` method or the functions used within `.run()`.

## Key components of the `songpipe` package

### How the `CalibrationSet` class works
The `songpipe.calib.CalibrationSet` class represents a collection everything we need in order to extract a science image for a given instrument mode. Once the `CalibrationSet` is ready, we can extract science images in that mode by feeding them to the `.extract()` method of the `CalibrationSet` instance.

When creating a `CalibrationSet` for an instrument mode, it requires a list of bias/dark-corrected images that have been cropped and rotated to the default left-right orientation. We refer to these images as "prepared" and represent them as `Image` objects (see below). From this list, the `CalibrationSet` will pick the appropriate flats and wavelength calibrations, by inquiring each `Image` about its mode. We can then execute the desired calibration steps by calling methods of the `CalibrationSet`, e.g. `.trace_orders()`.

```py3
# Example, setting up a CalibrationSet for mode F1
calibration_set = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F1")
calibration_set.combine_flats()
calibration_set.trace_orders()
calibration_set.normalize_flat()
```
_Note: ThAr Calibrations are handled more explicitly (see runscripts), and then packaged back into the `CalibrationSet`._

The `MultiFiberCalibrationSet` is a subclass of `CalibrationSet`, designed to handle the SONG-Australia setup. A `MultiFiberCalibrationSet` links to a normal `CalibrationSet` for each fiber, such that it can use the individual order trace and extract the two interlacing spectra into separate output files.

```py3
# Example, linking modes F1 and F2 to the F12 multi-fiber calibration set
calibs = {}
calibs['F1'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F1")
calibs['F2'] = CalibrationSet(prep_images, calibdir, config, mask, instrument, "F2")
calibs['F12'] = MultiFiberCalibrationSet(prep_images, calibdir, config, mask, instrument, "F12")
calibs['F12'].link_single_fiber_calibs(calibs['F1'], calibs['F2'])
```

### `Image` and `Spectrum` classes
Mention also `ImageClass` and `SpectrumClass` 

### Runscript structure
