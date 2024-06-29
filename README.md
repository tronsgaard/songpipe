# SONG pipeline
`songpipe` is the new default extraction pipeline for [SONG](https://phys.au.dk/song/) (Stellar Observations Network Group), currently supporting the Tenerife and Australia nodes. 

`songpipe` runs in Python (version 3.8 or higher) and works as a wrapper around the core routines of [PyReduce](https://github.com/AWehrhahn/PyReduce/) ([Piskunov & Valenti (2002)](https://doi.org/10.1051/0004-6361:20020175), [Piskunov, Wehrhahn & Marquart (2020)](https://doi.org/10.1051/0004-6361/202038293)). As PyReduce is not being actively maintained at the moment (April 2024), and some adjustments of the code were necessary, `songpipe` uses a fork of the PyReduce repository, located [here](https://github.com/tronsgaard/PyReduce).

`songpipe` is still being developed (see below).

## Installation
- Create and enter a virtual environment (`venv /path/to/new/virtual/environment`)
- Download the modified version of `PyReduce` (https://github.com/tronsgaard/PyReduce) and install it using `pip install --editable .` from within the folder (the `--editable` flag can be omitted, but it makes it easier to update the code).
- Download the latest release of `songpipe` or simply checkout this repository (`git checkout https://github.com/tronsgaard/songpipe.git`).
- Install required packages listed in [requirements.txt](requirements.txt).
  - Note: Packages `dill` and `colorlog` are optional and only required for FITS header caching and colored terminal output, respectively.

## Running
`songpipe` runs from the terminal, just execute `run_australia.py` or `run_tenerife.py` located in the root of the repository. For full syntax, call the script with `--help`.

The scripts must be called with a date string (`YYYYMMDD`) as the first argument. The keywords `--rawdir DIRPATH` and `--outdir DIRPATH` speficiy the location of the input and output directories. If called with `--confirm`, the script will pause and ask the user to confirm the settings and directory structure before proceeding.

## Known issues
- `astropy.io.fits` regularly complains about truncated FITS files, but no actual issues with the files have been found.
- Extraction of single files using `--extract` keyword is currently not working as expected.

## Planned development
- Include subtraction of scattered light
- Include order curvature for Tenerife
- Include bad pixel mask
- Support pre-CMOS Tenerife data
- Multiprocessing support
- Automatic deletion of intermediate files (`prep/` directory)
- Easier installation with pip (create setup.py)

## Further reading
- A more detailed description about [how `songpipe` works](docs/how_songpipe_works.md)
- [`PyReduce` documentation](https://pyreduce-astro.readthedocs.io/en/latest/)
