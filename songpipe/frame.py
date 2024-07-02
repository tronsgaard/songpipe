from os.path import basename
import numpy as np

from . import config
from .misc import construct_filename, apply_limit, header_insert

import logging
logger = logging.getLogger(__name__)

"""
This module contains the Frame and FrameList classes, on which the Image/ImageList and Spectrum/SpectrumList classes are based.
"""

class Frame:
    """
    Base class for Image and Spectrum. 
    Parent-child structure allows for HighLowImage containing two Image objects
    """

    def __init__(self):
        raise NotImplementedError  # Frame class should not be used directly

    def get_header_value(self, key):
        """If header does not contain the key, go back and check the parent frame (e.g. a HighLowImage)"""
        try:
            return self.header[key]
        except KeyError:
            if self.parent is not None:
                return self.parent.get_header_value(key)
            else:
                raise

    def header_insert(self, key, value=None, comment=''):
        """Insert or set FITS header keyword"""
        self._header = header_insert(self._header, key, value, comment)

    @property
    def header(self):
        return self._header

    @property
    def object(self):
        return self.get_header_value('OBJECT')

    @property
    def exptime(self):
        return self.get_header_value('EXPTIME')

    @property
    def date_start(self):
        return self.get_header_value('DATE-OBS')
    
    @property
    def jd_start(self):
        return self.get_header_value('JD-DATE')
    
    @property
    def mjd_start(self):
        return self.get_header_value('MJD-DATE')

    @property
    def mjd_mid(self):
        """Get MJD_MID from header or calculate from exptime"""
        try:
            return self.get_header_value('MJD-MID')
        except KeyError:
            exptime_days = self.exptime/86400.
            return self.mjd_start + 0.5*exptime_days

    @property
    def type(self):
        return self.get_header_value('IMAGETYP')
    
    @property
    def observatory(self):
        """Get observatory name from FITS header"""
        try:
            return self.get_header_value('OBSERVAT')
        except KeyError:
            logger.warning(f'Missing keyword OBSERVAT: {self.filename}')
            return 'Unknown'

    @property
    def is_australia(self):
        """Check if this frame comes from Australia"""
        return True if self.observatory == 'Mt. Kent' else False
    
    @property
    def is_tenerife(self):
        """Check if this frame comes from Tenerife"""
        return True if self.observatory in ('Observatorio del Teide', 'Tenerife') else False

    @property
    def mode(self):
        """
        Determine the instrument mode: 
           Mt. Kent:    F1, F2, F12, SLIT, DARK, UNKNOWN
           Tenerife:    SLITx, DARK, UNKNOWN
        """
        try:
            # First look for SONGPIPE header keyword
            return self.get_header_value('PL_MODE')
        except KeyError:
            # Otherwise, derive from rest of header
            if self.type in ('DARK', 'BIAS'):
                return 'DARK'
            # Tenerife:
            if self.is_tenerife:
                slit = self.get_header_value('SLIT')
                return f'SLIT{slit}'
            # Australia:
            if self.is_australia:
                if self.type == 'FLAT' and self.get_header_value('LIGHTP') == 1:
                    return 'SLIT'
                if self.type in ('FLAT', 'FLATI2', 'THAR', 'FP'):
                    # Check telescope shutters
                    tel1 = self.get_header_value('TEL1_S')
                    tel2 = self.get_header_value('TEL2_S')
                    if tel1 == 1 and tel2 == 1:
                        return 'F12'
                    if tel1 == 1:
                        return 'F1'
                    if tel2 == 1:
                        return 'F2'
            # In any other case (all observations):
            # Note: This includes all star spectra from Mt. Kent, as the header 
            #       doesn't tell if there is starlight in both fibres.
            return 'UNKNOWN'

    # Misc
    def construct_filename(self, **kwargs):
        """
        Wrapper for the `construct_filename()` method.
        Passes object name from FITS header, if config.OBJECT_IN_FILENAME is True
        """
        if 'object' in kwargs:
            obj = kwargs.pop('object')  # Removes item from kwargs
        elif config.OBJECT_IN_FILENAME is True:
            obj = self.object
        else:
            obj = None
        
        # Get original filename
        try:
            orig_filename = basename(self.get_header_value('FILE'))
        except KeyError:
            try:
                orig_filename = basename(self.get_header_value('ORIGFILE'))
            except KeyError:
                import re
                try:
                    pattern = r'(s\d_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}).*(\.fits.*)'
                    orig_filename = re.findall(pattern, self.filename)[0]
                except IndexError as e:
                    logger.error(f'Could not determine original filename: {self.filename}')
                    raise e
        return construct_filename(orig_filename, object=obj, **kwargs)

class FrameList:
    """ImageList and SpectrumList inherit shared properties and methods from this class"""
    
    @property
    def files(self):
        return [frame.filename for frame in self.frames]

    def __len__(self):
        """Provides len() compatibility"""
        return len(self.frames)

    def __iter__(self):
        """Allows the FrameList to be used directly in loops"""
        return self.frames.__iter__()

    def __getitem__(self, item):
        try:
            return self.frames.__getitem__(int(item))
        except ValueError:
            # If the int() cast fails
            pass
        # Get item by filename
        result = self.filter(filename_contains=item)
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            raise KeyError(f'Multiple files matching: "{item}"')
        else:
            raise KeyError(f'No such filename in list: {item}')

    def summarize(self):
        """Print summary of image types in the list"""
        count = {}
        for frame in self.frames:
            count[frame.type] = count.get(frame.type, 0) + 1

        for k, n in count.items():
            print(k, n)

    def get_exptimes(self, threshold=0.1, tol=0.1):
        """
        Return a list of exptimes used in the data
        
        Parameters: 
            threshold : exclude exptimes shorter than `threshold` (default: 0.1s)
            tol :       cluster similar exptimes in bins of width 2*tol
        
        """
        all_exptimes = np.array([frame.exptime for frame in self.frames])
        all_exptimes = all_exptimes[all_exptimes >= threshold]
        if tol > 0:
            #exptimes = np.unique(np.round((all_exptimes)/2/tol).astype(int))*2*tol  # cast as integer for faster computation
            exptimes, indices = np.unique(np.round((all_exptimes)/2/tol).astype(int), return_inverse=True)
            exptimes = exptimes.astype(float)
            # Find the median value within each bin
            for m in range(len(exptimes)):
                exptimes[m] = np.median(all_exptimes[indices==m])

        else:
            exptimes = np.unique(all_exptimes)
        return exptimes.tolist()

    def count(self, **kwargs):
        """Passes all arguments to self.filter() and counts the number of frames returned"""
        res = self.filter(**kwargs)
        return len(res)
    
    def append(self, frame):
        """
        Append `frame` to `self.frames`.
        If `frame` has a filename, this method will remove any 
        existing duplicates in `self.frames` before appending `frame`.
        """
        if frame.filename is not None:
            for i,f in enumerate(self.frames):
                if f.filename == frame.filename:
                    self.frames.pop(i)  # Remove old image object with matching filename
        self.frames.append(frame)

    def extend(self, frames):
        """Extend with a list of frames, calling self.append() for each item"""
        for frame in frames:
            self.append(frame)

    def __iadd__(self, other):
        """Enables the += operator"""
        self.extend(other)
        return self
    
    def list(self, add_keys=None, outfile=None, silent=False):
        """
        Print a pretty list of filenames and some fits keywords.
        Add list of keys by using add_keys=['KEY1', 'KEY2', ...]
        Print to txt file by setting outfile=<path/to/file>
        """
        if len(self) == 0:
            return

        if isinstance(add_keys, str):
            add_keys = [add_keys]
        elif add_keys is None:
            add_keys = []

        buffer = []  # To be filled with all data before printing
        widths = {}  # Column widths
        for frame in self.frames:
            d = {}  # Dictionary of data for this image
            if frame.filename is not None:
                d['filename'] = basename(frame.filename)
            else:
                d['filename'] = ''
            
            try:
                d['image_type'] = frame.type
            except KeyError:
                d['image_type'] = ''
            
            try:
                d['mode'] = frame.mode
            except KeyError:
                d['mode'] = ''

            try:
                d['exptime'] = frame.exptime
            except KeyError:
                d['exptime'] = ''

            # Display additional keywords
            for k in add_keys:
                try:
                    d[k] = frame.header[k]
                except KeyError:
                    d[k] = ''
            
            # Display object last
            try:
                d['object'] = frame.object
            except KeyError:
                d['object'] = ''
            
            # Append row to buffer
            buffer.append(d)
            # Update dict of column widths
            for k in d.keys():
                widths[k] = max((len(str(d[k])), widths.get(k, 0),))
        
        # Define headers and update widths
        headers = {k: k.title() for k in widths.keys()}
        headers['image_type'] = 'Type'
        headers['exptime'] = 'Exp'
        widths = {k:max((len(headers[k]), w)) for k,w in widths.items()}
        
        # Define format string
        fmt = '  '.join([f'{{{k}:<{w}}}' for k, w in widths.items()]) + '\n'
        
        # Generate header line
        lines = []
        lines.append(fmt.format(**headers))
        
        # Generate table rows
        for d in buffer:
            lines.append(fmt.format(**d))

        # Export to file if requeted
        if outfile is not None:
            try:
                with open(outfile, 'w') as h:
                    h.writelines(lines)
            except IOError as e:
                logger.error('Could not export nightlog to txt file. Continuing..')
                logger.error(e)
        # Print
        if silent is not True:
            print('------------------------')
            for l in lines:
                print(l, end='')
            print('------------------------')
            print(f'Total: {len(self.frames)}')
            print('------------------------')

    def filter(self, object_contains=None, object_exact=None, 
               filename_contains=None, filename_exact=None,
               image_type=None, image_type_exclude=None, 
               mode=None, mode_exclude=None,
               exptime=None, exptime_lte=None, exptime_tol=0.1, 
               limit=None):
        """
        Filter list by various criteria and return result as a new list

        Arguments:
            object_contains :   Filter by partial object name
            object_exact :      Filter by exact object name
            filename_contains : Filter by partial filename
            filename_exact :    Filter by exact filename
            image_type :        Filter by image type or list of image types (e.g. `BIAS`, `STAR`)
            image_type_exclude: Exclude image type or list of image types
            mode :              Filter by instrument mode or list of modes (e.g. `F1`)
            mode_exclude :      Exclude instrument mode or list of instrument modes
            exptime :           Filter by exposure time
            exptimel_lt :       Filter by exposure time less than
            exptime_tol :       Tolerance used for exposure time filter (default: 0.1 s)
            limit :             Limit number of frames returned (default: unlimited)
        """

        def _ensure_list(x):
            """Ensure that x is a list/iterable"""
            if isinstance(x, str) or not hasattr(x, '__iter__'):
                x = [x]
            return x
        
        mask = [True] * len(self)
        for k, im in enumerate(self.frames):
            if object_contains is not None and object_contains not in im.object:
                mask[k] = False
                continue
            if object_exact is not None and object_exact != im.object:
                mask[k] = False
                continue
            if filename_contains is not None and (im.filename is None or filename_contains not in im.filename):
                mask[k] = False
                continue
            if filename_exact is not None and (im.filename is None or filename_exact != im.filename):
                mask[k] = False
                continue
            if image_type is not None:
                image_type = _ensure_list(image_type)
                if im.type not in image_type:
                    mask[k] = False
                    continue
            if image_type_exclude is not None:
                image_type_exclude = _ensure_list(image_type_exclude)
                if im.type in image_type_exclude:
                    mask[k] = False
                    continue
            if mode is not None:
                mode = _ensure_list(mode)
                if im.mode not in mode:
                    mask[k] = False
                    continue
            if mode_exclude is not None:
                mode_exclude = _ensure_list(mode_exclude)
                if im.mode in mode_exclude:
                    mask[k] = False
                    continue
            if exptime is not None and np.abs(exptime - im.exptime) > exptime_tol:
                mask[k] = False
                continue
            if exptime_lte is not None and im.exptime > exptime_lte + exptime_tol:
                mask[k] = False
                continue
        # Apply mask
        frames = np.array(self.frames)[mask].tolist()
        # Apply limit
        frames = apply_limit(frames, limit)
        # Return frames as a native python list
        return frames
    
    def get_closest(self, mjd):
        """Given a MJD time, return the frame with the closest MJD mid time"""
        # Build list of tuples (frame, timediff)
        res = []
        for frame in self.frames:
            d = mjd - frame.mjd_mid
            res.append((frame, d))
        # Sort according to abs(timediff)
        res = sorted(res, key=lambda x: np.abs(x[1]))
        frame, d = res[0]
        # Log nicely
        if np.abs(d) < 1/24:
            logger.debug(f'Found frame closest in time: {24*60*d:.2f} minutes')
        elif np.abs(d) < 1:
            logger.debug(f'Found frame closest in time: {24*d:.2f} hours')
        else:
            logger.debug(f'Found frame closest in time: {d:.2f} days')
        if frame.filename is not None:
            logger.debug(f'Filename: {frame.filename}')
        return frame
