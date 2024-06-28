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

In `songpipe` we handle the calls to `PyReduce` inside a class called `CalibrationSet`, which is described below. We create the `Step` objects and call either their `.run()` methods or directly the functions used by `.run()`.

## How the `CalibrationSet` class works
The `songpipe.calib.CalibrationSet` class represents a collection everything we need to extract a science image for a given instrument mode. 

Once the `CalibrationSet` is ready, we can extract science images in that mode by feeding them to the `.extract()` method of the `CalibrationSet` instance.

## `Image` and `Spectrum` classes

## Runscript structure
