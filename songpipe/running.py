import sys
from os.path import join, dirname, exists
from os import makedirs
import argparse

import songpipe

import logging
from logging.handlers import RotatingFileHandler
import tqdm

logger = logging.getLogger(__name__)

def parse_arguments(defaults):
    """
    This function parses the supplied command line arguments 
    using argparse and returns the `opts` object
    """

    # Set up command line arguments
    ap = argparse.ArgumentParser()
    # Directory structure
    ap.add_argument('datestr', metavar='date_string', type=str, default=None,
                    help='Night date (as a string), e.g. `20220702`')
    ap.add_argument('--basedir', type=str, default=defaults['basedir'],
                    help=f'Base directory (default: {defaults["basedir"]})')
    ap.add_argument('--rawdir', type=str, default=None,
                    help=f'Specify raw directory (default: <basedir>/star_spec/<date_string>/raw)')
    ap.add_argument('--outdir', type=str, default=None,
                    help=f'Specify raw directory (default: <basedir>/extr_spec/<date_string>)')
    ap.add_argument('--calibdir', type=str, default=None,
                    help=f'Specify calib directory (default: <basedir>/extr_spec/<date_string>/calib)')
    ap.add_argument('--logdir', type=str, default=None,
                    help=f'Specify log directory (default: <basedir>/extr_spec/<date_string>/log)')
    # Actions
    ap.add_argument('--debug', action='store_true',
                    help='Set log level to debug (log everything)')
    ap.add_argument('--plot', action='store_true',
                    help='Activate plotting in PyReduce')
    ap.add_argument('--reload-cache', action='store_true',
                    help='Ignore cached FITS headers and reload from files')
    ap.add_argument('--simple-extract', action='store_true',
                    help='Extract using simple summation across orders (faster than optimal extraction)')
    ap.add_argument('--silent', action='store_true',
                    help='Silent mode (useful when running in background)')
    ap.add_argument('--skip-flati2', action='store_true',
                    help='Skip extraction of FLATI2 spectra')
    ap.add_argument('--skip-fp', action='store_true',
                    help='Skip extraction of Fabry Perót (FP) spectra')
    # TODO:
    #ap.add_argument('--ignore-existing', action='store_true',
    #                help='Ignore existing output files and run extraction again')

    opts = ap.parse_args()
    if opts.silent:
        # Silence terminal output by redirecting stdout to /dev/null
        devnull = open('/dev/null', 'w')
        sys.stdout = devnull
        sys.stderr = devnull  # tqdm progress bars are printed through stderr; pyreduce has no option to silence tqdm
    if opts.rawdir is None:
        # Default to <basedir>/star_spec/<date_string>/raw
        opts.rawdir = join(opts.basedir, 'star_spec', opts.datestr, 'raw')
    if opts.outdir is None:
        # Default to <basedir>/extr_spec/<date_string>
        opts.outdir = join(opts.basedir, 'extr_spec', opts.datestr)
        makedirs(opts.outdir, exist_ok=True)
    if opts.calibdir is None:
        # Default to <basedir>/extr_spec/<date_string>/calib
        opts.calibdir = join(opts.outdir, 'calib')
        makedirs(opts.calibdir, exist_ok=True)
    if opts.logdir is None:
        # Default to <basedir>/extr_spec/<date_str>/log
        opts.logdir = join(opts.outdir, 'log')
    
    return opts


# Logging
class TqdmLoggingHandler(logging.Handler):
    """
    This logging handler prints log messages through the tqdm.write() function, 
    ensuring that we don't break the tqdm progress bars (copied from pyreduce).
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def setup_logger(log_file, name=None, silent=False, reset=True, debug=False):
    """
    name=None sets up the root logger
    """

    # Retrieve named logger and set log level
    logger = logging.getLogger(name)
    if debug is True:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logging.captureWarnings(True)

    # Reset logging handlers (pyreduce adds its own handlers when imported)
    if reset is True:
        logger.handlers.clear()

    # Set up log file
    makedirs(dirname(log_file), exist_ok=True)
    log_file_exists = exists(log_file)  # Check if the log file already exists, if yes, we'll rename it (rotate it) in a second
    file_handler = RotatingFileHandler(log_file, backupCount=10)  # RotatingFileHandler ensures that most recent old logs are renamed and preserved
    if log_file_exists:
        file_handler.doRollover()  # This renames the old log file(s) and creates a new file in its place
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')  # Log columns
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Set up console log
    if not silent:
        #console_handler = logging.StreamHandler()  # Not compatible with tqdm progress bars
        console_handler = TqdmLoggingHandler()

        # Add colors if available
        try:
            import colorlog
            console_formatter = colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(message)s")
        except ImportError:
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            print("Install colorlog for colored logging output")

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def log_summary(opts, image_class):
    """
    Log various parameters and software versions
    """
    logger.info('SONG pipeline starting..')
    logger.info('------------------------')
    logger.info(f'Python version:    {sys.version.split(" ")[0]}')
    logger.info(f'songpipe version:  {songpipe.__version__}')
    logger.info('------------------------')
    logger.info(f'Raw directory:     {opts.rawdir}')
    logger.info(f'Output directory:  {opts.outdir}')
    logger.info(f'Calib directory:   {opts.calibdir}')
    logger.info(f'Log directory:     {opts.logdir}')
    logger.info('------------------------')
    logger.info(f'Plotting:          {opts.plot}')
    logger.info(f'Reload cache:      {opts.reload_cache}')
    logger.info(f'Simple extraction: {opts.simple_extract}')
    logger.info(f'Silent:            {opts.silent}')
    logger.info('------------------------')

    logger.info(f'Image class: <{image_class.__module__}.{image_class.__name__}>')
    logger.info('------------------------')


def load_images(filemask, image_class, reload_cache=False, outdir=dirname(__name__), silent=False):
    """
    Load all FITS headers as Image objects
    Objects are saved to a dill file called .songpipe_cache, saving time if we need to run the pipeline again
    """

    cachefile = join(outdir, ".songpipe_cache")

    images = None
    if reload_cache is False:
        try:
            import dill  # Similar to, yet better than, pickle
            with open(cachefile, 'rb') as h:
                images, version = dill.load(h)
            if version != songpipe.__version__:
                logger.warning("Cache version mismatch.")
                images = None
            else:
                logger.info(f'Loaded FITS headers from cache: {cachefile}')
        except ImportError:
            logger.info('Install dill to enable caching of FITS headers.')
        except (FileNotFoundError,):
            pass
        except Exception as e:
            logger.warning(e)
            logger.warning('Could not reload FITS headers from cache')
    # If images is still None, it means we need to load the FITS headers from their source
    if images is None:
        logger.info('Loading FITS headers from raw images...')
        # The following line loads all *.fits files from the raw directory
        images = songpipe.ImageList.from_filemask(filemask, image_class=image_class, silent=silent)
        try:
            # Save objects for next time
            import dill
            with open(cachefile, 'wb') as h:
                dill.dump((images, songpipe.__version__), h)
        except Exception as e:
            logger.warning(e)
            logger.warning('Could not save cache. Continuing...')

    # Print and store list of observations
    images.list(outfile=join(outdir, '000_list.txt'), silent=silent)

    return images