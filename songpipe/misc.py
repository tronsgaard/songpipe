from os.path import splitext, dirname, exists
from os import makedirs
import astropy.io.fits as fits
import logging
from logging.handlers import RotatingFileHandler
import tqdm

logger = logging.getLogger(__name__)

"""
This module contains various functions, e.g. FITS header manipulation, to be used by the other modules
"""

def construct_filename(orig_filename, object=None, prepared=False, extracted=False, mode=None,
                       prefix=None, suffix=None, ext=None):
    """Construct a standardized filename based on the properties of the file"""
    filename, old_ext = splitext(orig_filename)
    if object is not None:
        filename += '_' + object.replace(' ', '_')
    # Prepend stuff
    if prefix is not None:
        filename = prefix + '_' + filename
    # Append stuff
    if mode is not None:
        filename += '_' + mode
    if prepared:
        filename += '_prep'
    elif extracted:
        filename += '_extr'
    if suffix is not None:
        filename += '_' + suffix
    if ext == None:
        if old_ext != '':
            ext = old_ext
        else:
            ext = '.fits'
    return filename + ext


def header_insert(hdr, key, value=None, comment=''):
    """Keep the header organized by grouping all pipeline keywords in a section"""
    hdr = hdr.__copy__()
    SECTION_HEADER = ('---PL---', '----PIPELINE----', '-------------------------------------')
    SECTION_FOOTER = ('--------', '----------------', '-------------------------------------')
    get_keys = lambda: list(hdr.keys())  # Get updated list of keys from header
    try:
        start = get_keys().index(SECTION_HEADER[0])
    except ValueError:
        hdr.set(*SECTION_HEADER)
        start = get_keys().index(SECTION_HEADER[0])
        hdr.insert(start, SECTION_FOOTER, after=True)
    end = start

    # Determine end of section
    in_section = True
    keys = get_keys()
    while in_section is True:
        if end + 1 >= len(keys) or keys[end + 1][0] == '-' or keys[end + 1] == 'COMMENT':
            in_section = False
        else:
            end += 1

    # Insert header key/value
    if key in get_keys():
        hdr.set(key, value, comment)
    else:
        hdr.insert(end, (key, value, comment), after=True)
    return hdr

def sanitize_header(hdr):
    """Remove duplicate key/value pairs and warn of duplicate keys with different values"""
    
    keys = list(hdr.keys())
    values = list(hdr.values())
    comments = list(hdr.comments)

    new_header = fits.Header()
    count_discarded = 0
    for i in range(len(hdr)):
        key, value, comment = keys[i], values[i], comments[i]
        
        if key not in new_header: 
            # If key not already in header
            new_header.append((key, value, comment))

        elif key in new_header and value != new_header[key]:
            # If key already in header, but different value (warn the user)
            logger.warning(f'Conflicting header values: {key}: "{value}" vs. "{new_header[key]}"')
            new_header.append((key, value, comment))

        else:
            # Otherwise, don't add to header (key/value pair matches existing)
            count_discarded += 1
    logger.info(f'{count_discarded} key/value pairs removed.')
    return new_header
         

def apply_limit(array, limit):
    """SQL-like limit syntax"""
    if not hasattr(limit, '__iter__'):
        limit = (limit,)
    return array[slice(*limit)]

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

def setup_logger(log_file, name=None, level=logging.INFO, silent=False, reset=True):
    """
    name=None sets up the root logger
    """

    # Retrieve named logger and set log level
    logger = logging.getLogger(name)
    logger.setLevel(level)
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
