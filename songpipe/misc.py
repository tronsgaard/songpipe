from os.path import splitext, dirname, exists
from os import makedirs
import astropy.io.fits as fits

from logging import getLogger
logger = getLogger(__name__)

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

