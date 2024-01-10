"""Tools for organizing, analyzing, and selecting dark frames"""

from .image import ImageList

from os.path import join, exists
from os import makedirs

import logging
logger = logging.getLogger(__name__)

DEFAULT_EXPTIME_TOL = 0.1

class DarkManager(ImageList):
    """Find and manage available dark frames"""
    
    def __init__(self, images, image_class=None, savedir=None,
                 min_dark_images=1, min_bias_images=1, 
                 combine_method='median', ):
        """
        Arguments:
            images :          Already prepared darks (list or ImageList)
            image_class :     Image or HighLowImage
            savedir :         Where to save combined darks
            min_dark_images : Minimal number of frames to combine for darks
            min_bias_images : Minimal number of frames to combine for bias
            combine_method :  Method used for combining frames (default: median)
        """
        super().__init__(images, image_class=image_class)
        self.savedir = savedir
        self.min_dark_images = min_dark_images
        self.min_bias_images = min_bias_images
        self.combine_method = combine_method
        logger.info('DarkManager initialized.')
        logger.warning('"The fear of loss is a path to the dark side." -Yoda')  # Set the right mood
    
    def construct_filename(self, exptime):
        """Determines the filename for an n-second master dark"""
        filename = f'master_dark_{exptime:.0f}s.fits'
        if self.savedir is not None:   # Savedir should always be set though..
            return join(self.savedir, filename)
        return filename
    
    def build_master_bias(self, images, silent=False):
        """Given a list if images, build a master bias"""

        outfile = join(self.savedir, 'master_bias.fits')
        if exists(outfile):
            # Load master bias with proper image class
            logger.info(f'Master bias already exists - loading from {outfile}')
            master_bias = self.image_class(filename=outfile)
        else:
            logger.info('Assembling master bias...')
            # Fetch a list of bias images
            bias_list = images.filter(image_type='BIAS')
            bias_list.list()
            # Check that we have enough images
            if len(bias_list) < self.min_bias_images:
                logger.warning(f'Not enough bias images. \
                                 Expected {self.min_bias_images}, found {len(bias_list)}')
                return

            # Assemble master bias using specified combine function
            master_bias = bias_list.combine(method=self.combine_method, silent=silent)
            master_bias.save_fits(outfile, overwrite=True, dtype='float32')

        # Add bias to master list and return
        self.append(master_bias)
        return master_bias

    
    def build_master_dark(self, images, exptime, exptime_tol=DEFAULT_EXPTIME_TOL, silent=False):
        """Given a list of images, build a master dark with a certain exptime"""
        
        outfile = self.construct_filename(exptime)

        if exists(outfile):
            logger.info(f'Master dark ({exptime:.0f}s) already exists - loading from {outfile}')
            # Load master dark with proper image class
            master_dark = self.image_class(filename=outfile)
        else:
            logger.info(f'Assembling {exptime} s master dark')
            dark_list = images.filter(exptime=exptime, exptime_tol=exptime_tol, image_type='DARK')
            # Assert whether we have enough images
            if len(dark_list) < self.min_dark_images:
                logger.warning(f'Not enough dark images for {exptime} s master dark. \
                                 Expected {self.min_dark_images}, found {len(dark_list)}')
                return

            # Assemble master dark using the specified combine function
            master_bias = self.get_master_bias()
            master_dark = dark_list.combine(method=self.combine_method, silent=silent)
            master_dark.subtract_bias(master_bias, inplace=True)  # Important!
            master_dark.save_fits(outfile, overwrite=True, dtype='float32')
    
        # Add to master list and return
        self.append(master_dark)
        return master_dark
    
    def build_all_master_darks(self, images, exptime_tol=DEFAULT_EXPTIME_TOL, silent=False):
        """Given a list of images, build all the master darks"""

        images = images.filter(image_type='DARK')
        exptimes = images.get_exptimes(tol=exptime_tol)
        # Log
        logger.debug('Available dark exptimes:')
        for exptime in exptimes:
            logger.debug(images.count(exptime=exptime, exptime_tol=exptime_tol))
        
        logger.info('Now building master darks...')
        for exptime in exptimes:
            self.build_master_dark(images, exptime, exptime_tol=exptime_tol, silent=silent)

    def check_exptimes(self, exptimes, exptime_tol=DEFAULT_EXPTIME_TOL, min_exptime=0):
        """Given a list of exptiems, check that we have darks for all"""
        missing = False
        for exptime in exptimes:
            # Ignore short exptimes
            if exptime <= min_exptime:
                continue
            # Try to get master dark for each exptime
            try:
                self.get_master_dark(exptime, exptime_tol=exptime_tol)
            except MissingDarkError:
                logger.error(f'Master dark missing for exptime {exptime} s')
                missing = True
        if missing is True:
            raise MissingDarkError()
        else:
            logger.info(f'Master darks exist for all exptimes > {min_exptime} s')

    def get_master_dark(self, exptime, mjd=None, 
                        exptime_tol=DEFAULT_EXPTIME_TOL, 
                        imtype='DARK'):
        """
        Return a master dark with matching exptime. 
        If multiple matches, pick closest in time.
        If no exptime match, default to closest exptime below.
        """
        
        # First we list all darks and check that list is not empty
        all_master_darks = self.filter(image_type=imtype)
        if len(all_master_darks) == 0:
            raise MissingDarkError(f'No master darks available ({imtype})')

        # Then we look for an exact match to exptime
        darks_exact = all_master_darks.filter(exptime=exptime, exptime_tol=exptime_tol)
        if len(darks_exact) == 1:
            return darks_exact[0]
        elif len(darks_exact) > 1:
            if mjd is not None:
                # Get closest in time
                return darks_exact.get_closest(mjd)
            else:
                # If no reference time is provided, pick the last added
                return darks_exact[-1]
        
        # If we reached this far, it means that no master darks were found with exact exptime
        # Get a list of all master darks with shorter exptime and pick the longest
        darks_lte = all_master_darks.filter(exptime_lte=exptime, exptime_tol=exptime_tol)
        if len(darks_lte) == 0:
            raise MissingDarkError(f'No master darks available with exptime <= {exptime} s')
        exptimes = darks_lte.get_exptimes()
        new_exptime = max(exptimes)
        logger.warning(f'No master dark available with exptime {exptime} s. Using {new_exptime} s instead.')
        return self.get_master_dark(new_exptime, mjd=mjd, exptime_tol=exptime_tol, imtype=imtype)



    def get_master_bias(self, mjd=None):
        """Return the master bias, if there are multiple, pick the closest in time"""
        return self.get_master_dark(0, mjd=mjd, imtype='BIAS')



class NotEnoughImagesError(Exception):
    pass

class MissingDarkError(Exception):
    pass
