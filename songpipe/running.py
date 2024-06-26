"""Tools for running songpipe"""

import sys
from os.path import join, dirname, exists
from os import makedirs
import argparse

from .image import ImageList

import logging
from logging.handlers import RotatingFileHandler
import tqdm

logger = logging.getLogger(__name__)

def parse_arguments(basedir):
    """
    This function parses the supplied command line arguments 
    using argparse and returns the `opts` object
    """

    # Set up command line arguments
    ap = argparse.ArgumentParser()
    # Directory structure
    ap.add_argument('datestr', metavar='date_string', type=str, default=None,
                    help='Night date (as a string), e.g. `20220702`')
    ag = ap.add_argument_group('Directories')
    ag.add_argument('--basedir', type=str, default=basedir, metavar='DIRPATH', 
                    help=f'Base directory (default: {basedir})')
    ag.add_argument('--rawdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify raw directory (default: <basedir>/star_spec/<date_string>/raw)')
    ag.add_argument('--outdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify raw directory (default: <basedir>/extr_spec/<date_string>)')
    ag.add_argument('--darkdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify dark directory (default: <outdir>/dark)')
    ag.add_argument('--prepdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify prep directory (default: <outdir>/prep)')
    ag.add_argument('--calibdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify calib (flat/trace) directory (default: <outdir>/calib)')
    ag.add_argument('--thardir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify ThAr directory (default: <outdir>/thar)')
    ag.add_argument('--fpdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify Fabry-Perót directory (default: <outdir>/fp)')
    ag.add_argument('--flati2dir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify flatI2 directory (default: <outdir>/flati2)')
    ag.add_argument('--stardir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify star directory (default: <outdir>/star)')
    ag.add_argument('--logdir', type=str, default=None, metavar='DIRPATH', 
                    help=f'Specify log directory (default: <outdir>/log)')
    # Calibrations
    ag = ap.add_argument_group('Calibration')
    ag.add_argument('--calib-only', action='store_true',
                    help='Exit after reducing calibs and ThAr')
    ag.add_argument('--ignore-darks', action='store_true',
                    help='Ignore darks from this night (add darks from other nights using --add-darks)')
    ag.add_argument('--add-darks', type=str, default=[], action='append', metavar='DIRPATH',
                    help=f'Specify directory with additional master darks (repeated use allowed)')

    ag.add_argument('--ignore-flats', action='store_true',
                    help='Ignore flats from this night (copy calibs from another night using --copy-calibs)')
    ag.add_argument('--copy-calibs', type=str, default=None, metavar='DIRPATH',
                    help=f'Copy flat calibrations (trace, normflat, scatter) from this directory')

    ag.add_argument('--ignore-thars', action='store_true',
                    help='Ignore ThAr calibs from this night (add ThAr calibs from other nights using --add-thars)')
    ag.add_argument('--add-thars', type=str, default=[], action='append', metavar='DIRPATH',
                    help=f'Specify directory with additional ThAr calibs (repeated use allowed)')
    # Exctraction
    ag = ap.add_argument_group('Extraction')
    ag.add_argument('--simple-extract', action='store_true',
                    help='Extract using simple summation across orders (faster than optimal extraction)')
    ag.add_argument('--skip-flati2', action='store_true',
                    help='Skip extraction of flatI2 spectra')
    ag.add_argument('--skip-fp', action='store_true',
                    help='Skip extraction of Fabry Perót (FP) spectra')
    ag.add_argument('--extract', action='store', metavar='FILEPATH',
                    help='Path to a single file to extract (prep file)')
    # Actions
    ag = ap.add_argument_group('Actions')
    ag.add_argument('--confirm-settings', action='store_true',
                    help='Pause script and wait for user to review and confirm settings')
    ag.add_argument('--plot', action='store_true',
                    help='Activate plotting in PyReduce')
    ag.add_argument('--reload-cache', action='store_true',
                    help='Ignore cached FITS headers and reload from files')
    ag.add_argument('--skip-obslog', action='store_true',
                    help='Don\'t save text file with list of observations')
    ag.add_argument('--obslog-only', action='store_true',
                    help='Exit after loading files and storing obslog')
    ag.add_argument('--silent', action='store_true',
                    help='Silent mode (useful when running in background)')
    ag.add_argument('--debug', action='store_true',
                    help='Set log level to debug (log everything)')
    # TODO:
    #ap.add_argument('--ignore-existing', action='store_true',
    #                help='Ignore existing output files and run extraction again')

    opts = ap.parse_args()
    if opts.silent:
        # Silence terminal output by redirecting stdout to /dev/null
        devnull = open('/dev/null', 'w')
        sys.stdout = devnull
        sys.stderr = devnull  # tqdm progress bars are printed through stderr; pyreduce has no option to silence tqdm
    
    # Set default directories
    if opts.rawdir is None:
        # Default to <basedir>/star_spec/<date_string>/raw
        opts.rawdir = join(opts.basedir, 'star_spec', opts.datestr, 'raw')
    if opts.outdir is None:
        # Default to <basedir>/extr_spec/<date_string>
        opts.outdir = join(opts.basedir, 'extr_spec', opts.datestr)
        makedirs(opts.outdir, exist_ok=True)
    if opts.darkdir is None:
        # Default to <outdir>/dark
        opts.darkdir = join(opts.outdir, 'dark')
        makedirs(opts.darkdir, exist_ok=True)
    if opts.prepdir is None:
        # Default to <outdir>/prep
        opts.prepdir = join(opts.outdir, 'prep')
        makedirs(opts.prepdir, exist_ok=True)
    if opts.calibdir is None:
        # Default to <outdir>/calib
        opts.calibdir = join(opts.outdir, 'calib')
        makedirs(opts.calibdir, exist_ok=True)
    if opts.thardir is None:
        # Default to <outdir>/thar
        opts.thardir = join(opts.outdir, 'thar')
        makedirs(opts.thardir, exist_ok=True)
    if opts.stardir is None:
        # Default to <outdir>/star
        opts.stardir = join(opts.outdir, 'star')
        makedirs(opts.stardir, exist_ok=True)
    if opts.fpdir is None:
        # Default to <outdir>/fp
        opts.fpdir = join(opts.outdir, 'fp')
        makedirs(opts.fpdir, exist_ok=True)
    if opts.flati2dir is None:
        # Default to <outdir>/flati2
        opts.flati2dir = join(opts.outdir, 'flati2')
        makedirs(opts.flati2dir, exist_ok=True)
    if opts.logdir is None:
        # Default to <outdir>/log
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
    from . import __version__ as songpipe_version
    from pyreduce import __version__ as pyreduce_version

    logger.info('SONG pipeline starting..')
    logger.info(f'Called as: {" ".join(sys.argv)}')
    logger.info('------------------------')
    logger.info(f'Python version:    {sys.version.split(" ")[0]}')
    logger.info(f'songpipe version:  {songpipe_version}')
    logger.info(f'pyreduce version:  {pyreduce_version}')
    logger.info('------------------------')
    logger.info(f'Raw directory:     {opts.rawdir}')
    logger.info(f'Output directory:  {opts.outdir}')
    logger.info('------------------------')
    logger.info(f'Dark directory:    {opts.darkdir}')
    logger.info(f'Prep directory:    {opts.prepdir}')
    logger.info(f'Calib directory:   {opts.calibdir}')
    logger.info(f'ThAr directory:    {opts.thardir}')
    logger.info(f'Star directory:    {opts.stardir}')
    if opts.skip_fp is False:
        logger.info(f'FP directory:      {opts.fpdir}')
    if opts.skip_flati2 is False:
        logger.info(f'FlatI2 directory:  {opts.flati2dir}')
    logger.info(f'Log directory:     {opts.logdir}')
    logger.info('------------------------')
    logger.info(f'Calib only:        {opts.calib_only}')
    logger.info(f'Ignore darks:      {opts.ignore_darks}')
    logger.info(f'Add master darks:  {opts.add_darks}')
    logger.info(f'Ignore flats:      {opts.ignore_flats}')
    logger.info(f'Copy flat calibs:  {opts.copy_calibs}')
    logger.info(f'Ignore ThAr calibs:{opts.ignore_thars}')
    logger.info(f'Add ThAr calibs:   {opts.add_thars}')
    logger.info('------------------------')
    logger.info(f'Simple extraction: {opts.simple_extract}')
    logger.info(f'Skip FLATI2:       {opts.skip_flati2}')
    logger.info(f'Skip Fabry Pérot:  {opts.skip_fp}')
    logger.info(f'Extract:           {opts.extract}')
    logger.info('------------------------')
    logger.info(f'Plotting:          {opts.plot}')
    logger.info(f'Reload cache:      {opts.reload_cache}')
    logger.info(f'Skip obslog:       {opts.skip_obslog}')
    logger.info(f'Obslog only:       {opts.obslog_only}')
    logger.info(f'Silent:            {opts.silent}')
    logger.info(f'Debug mode:        {opts.debug}')
    logger.info('------------------------')
    logger.info(f'Image class: <{image_class.__module__}.{image_class.__name__}>')
    logger.info('------------------------')

    if opts.confirm_settings is True:
        input('Press ENTER to continue or Ctrl-C to abort...')


def load_images(filemask, image_class, ignore_list=None, reload_cache=False, outdir=dirname(__name__), silent=False):
    """
    Load all FITS headers as Image objects
    Objects are saved to a dill file called .songpipe_cache, saving time if we need to run the pipeline again
    """

    from . import __version__ as songpipe_version

    cachefile = join(outdir, ".songpipe_cache")

    images = None
    if reload_cache is False:
        try:
            import dill  # Similar to, yet better than, pickle
            with open(cachefile, 'rb') as h:
                images, version = dill.load(h)
            if version != songpipe_version:
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
        images = ImageList.from_filemask(filemask, ignore_list=ignore_list, image_class=image_class, silent=silent)
        logger.info(f'Loaded {len(images)} images')
        try:
            # Save objects for next time
            logger.info(f'Saving cache file: {cachefile}')
            import dill
            with open(cachefile, 'wb') as h:
                dill.dump((images, songpipe_version), h)
        except Exception as e:
            logger.warning(e)
            logger.warning('Could not save cache. Continuing...')

    return images


def read_ignore_list(filepath):
    """Load ignore list"""
    ignore_list = []
    try:
        with open(filepath) as h:
            for line in h:
                try:
                    item = line.split(' ')[0]
                    ignore_list.append(item)
                except:
                    pass
    except FileNotFoundError:
        logger.info(f'Ignore list not found: {filepath}')
