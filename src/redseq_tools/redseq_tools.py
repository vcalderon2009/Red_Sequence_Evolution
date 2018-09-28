#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-09-26
# Last Modified: 2018-09-26
# Vanderbilt University
from __future__ import absolute_import, division, print_function
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, redseq_tools"]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
__all__        = ["RedSeq"]
"""
Utilities for reading in the ML outputs for this project.
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.ml    import ml_utils        as cmlu
from cosmo_utils.mock_catalogues import catls_utils as cmcu

import numpy as num
import os
import pandas as pd
import pickle
from   astropy.io import fits
from   astropy.table import Table

# Functions

class RedSeq(object):
    """
    Class for manipulating the dataset.
    """
    def __init__(self, **kwargs):
        """
        This is the initial function for the `RedSeq` class object.

        Parameters:
        -----------
        """
        super().__init__()
        ## Assigning variables
        # Downloading dataset
        self.mband_1        = kwargs.get('mband_1', 'MAG_AUTO_G')
        self.mband_2        = kwargs.get('mband_2', 'MAG_AUTO_Z')
        self.mag_diff_tresh = kwargs.get('mag_diff_tresh', 4.)
        self.mag_min        = kwargs.get('mag_min', 24)
        self.mag_max        = kwargs.get('mag_max', 17)
        # Analysis
        self.radius         = kwargs.get('radius', 5./3600.)
        self.radius_unit    = kwargs.get('radius_unit', 'deg')
        self.cosmo_choice   = kwargs.get('cosmo_choice', 'WMAP7')
        self.zbin           = kwargs.get('zbin', 0.0125)
        self.zmin           = kwargs.get('zmin', 0.4)
        self.zmax           = kwargs.get('zmax', 1.0)
        self.input_loc      = kwargs.get('input_loc', 'RedMapper')
        # Extra variables
        self.proj_dict      = cwpaths.cookiecutter_paths(__file__)

    def input_catl_prefix_str(self, catl_kind='master'):
        """
        Prefix string for the main catalogues

        Parameters:
        ------------
        catl_kind : {'master', 'random', 'redmapper'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.
                - `random` : Catalogue of random catalogues
                - `redmapper` : Catalogue of RedMapper Clusters

        Returns
        ------------
        catl_pre_str : `str`
            Prefix string for the catalogue being analyzed
        """
        #
        # `catl_kind`
        if not (catl_kind in ['master', 'redmapper', 'random']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        # Prefix string
        catl_pre_arr = [catl_kind,
                        self.mband_1,
                        self.mband_2,
                        self.mag_diff_tresh,
                        self.mag_min,
                        self.mag_max]
        catl_pre_str = 'catl_{0}_{1}_{2}_{3}_{4}_{5}'
        catl_pre_str = catl_pre_str.format(*catl_pre_arr)

        return catl_pre_str

    def input_catl_file_path(self, catl_kind='master', check_exist=True,
        create_dir=False):
        """
        Path to the file being analyzed.

        Parameters:
        ------------
        catl_kind : {'master', 'random', 'redmapper'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.
                - `random` : Catalogue of random catalogues
                - `redmapper` : Catalogue of RedMapper Clusters

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ------------
        input_catl_dir : `str`
            Path to the file being analyzed.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        #
        # `catl_kind`
        if not (catl_kind in ['master', 'redmapper', 'random']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        # Path to directory
        input_catl_dir = os.path.join(  self.proj_dict['ext_dir'],
                                        catl_kind)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(input_catl_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(input_catl_dir)):
                msg = '`input_catl_dir` ({0}) was not found!'.format(
                    input_catl_dir)
                raise FileNotFoundError(msg)

        return input_catl_dir

    def input_catl_file(self, catl_kind='master', check_exist=True,
        ext='csv'):
        """
        Path to the file being analyzed.

        Parameters:
        ------------
        catl_kind : {'master', 'random', 'redmapper'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.
                - `random` : Catalogue of random catalogues
                - `redmapper` : Catalogue of RedMapper Clusters

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        ext : {'csv'} `str`
            Extension used for the catalogue file.

        Returns
        ------------
        input_file_path : `str`
            Path to the file being analyzed.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `catl_kind`
        if not (catl_kind in ['master', 'redmapper', 'random']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        ##
        ## Modifying extension
        if (catl_kind == 'master'):
            ext = 'csv'
        else:
            ext = 'fits'
        ##
        ## Path to the input file
        filepath = os.path.join(self.input_catl_file_path(catl_kind=catl_kind),
                            '{0}.{1}'.format(
                                self.input_catl_prefix_str(catl_kind=catl_kind),
                                ext))
        if check_exist:
            if not (os.path.exists(filepath)):
                msg = '`filepath` ({0}) was not found!'.format(
                    filepath)
                raise FileNotFoundError(msg)

        return filepath

    def extract_input_catl_data(self, catl_kind='master', ra_dec_only=True):
        """
        Extracts the data from the various sets of external/input files.

        Parameters:
        ------------
        catl_kind : {'master', 'random', 'redmapper'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.
                - `random` : Catalogue of random catalogues
                - `redmapper` : Catalogue of RedMapper Clusters

        ra_dec_only : `bool`, optional
            If True, only the `RA` and `DEC` columns will be used.
            This variable is set to `False` by default.

        Returns
        ------------
        catl_pd : `pd.DataFrame`
            DataFrame containing the `raw` data for a given `catl_kind` file.
        """
        # Check input parameters
        #
        # `catl_kind`
        if not (catl_kind in ['master', 'redmapper', 'random']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        #
        # `ra_dec_only`
        if not (isinstance(ra_dec_only, bool)):
            msg = '`ra_dec_only` ({1}) is not a valid type variable!'
            msg = msg.format(type(ra_dec_only))
            raise ValueError(msg)
        ## Path to external file
        catl_file_path = self.input_catl_file(catl_kind=catl_kind)
        ## Extracting data
        if (catl_kind == 'master'):
            # Reading in as CSV file and converting to pandas
            catl_pd = pd.read_csv(catl_file_path)
        else:
            ## Reading in FITS file and converting to pandas
            catl_fits_data = fits.getdata(catl_file_path)
            ## Converting to Astropy Table object
            catl_table_data = Table(catl_fits_data)
            catl_fits_data  = None
            # Selecting columns if necessary
            if ra_dec_only:
                # Columns to choose
                catl_table_data = catl_table_data['RA','DEC']
            # Converting to DataFrame
            catl_pd = catl_table_data.to_pandas()

        return catl_pd

    def catl_filtered_dir(self, catl_kind='master', check_exist=True,
        create_dir=False):
        """
        Path to the `filtered` file for `master`, after having applied all
        of the magnitude and redshift cuts.

        Parameters:
        ------------
        catl_kind : {'master'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ------------
        filtered_dir path : `str`
            Path to the `filtered` version of `master` file.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        #
        # `catl_kind`
        if not (catl_kind in ['master', 'redmapper', 'random']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        # Path to directory
        filtered_dir = os.path.join(self.proj_dict['int_dir'],
                                    'filtered',
                                    catl_kind)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(filtered_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(filtered_dir)):
                msg = '`filtered_dir` ({0}) was not found!'.format(
                    filtered_dir)
                raise FileNotFoundError(msg)

        return filtered_dir

    def catl_filtered_filepath(self, catl_kind='master', check_exist=True,
        ext='hdf5'):
        """
        Path to the `filtered` file for `master`, after having applied all
        of the magnitude and redshift cuts.

        Parameters:
        ------------
        catl_kind : {'master'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        ext : {'hdf5'}, optional
            Extension of th `filtered` catalogue file. This variable is set to
            'hdf5' by default.
        
        Returns
        ------------
        filtered_dir path : `str`
            Path to the `filtered` version of `master` file.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `catl_kind`
        if not (catl_kind in ['master']):
            msg = '`catl_kind` ({1}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        # Path to directory
        filtered_dir = self.catl_filtered_dir(catl_kind=catl_kind,
                        check_exist=False, create_dir=True)
        # Path to `filtered` catalogue
        catl_filt_path = os.path.join(filtered_dir,
                            '{0}_{1}.{2}'.format(
                                filtered_dir,
                                self.input_catl_prefix_str(catl_kind=catl_kind),
                                ext))
        # Check for its existence
        if check_exist:
            if not (os.path.exists(catl_filt_path)):
                msg = '`catl_filt_path` ({0}) was not found!'.format(
                    catl_filt_path)
                raise FileNotFoundError(msg)

        return catl_filt_path

    def extract_filtered_data(self, catl_kind='master', ext='hdf5',
        remove_file=False, return_pd=True):
        """
        Extracts the data for the `filtered` data.

        Parameters:
        ------------
        catl_kind : {'master'}, `str`
            Option for which kind of catalogue is being analyzed.
            Options:
                - `master` : Catalogue of objects.

        ext : {'hdf5'}, optional
            Extension of th `filtered` catalogue file. This variable is set to
            'hdf5' by default.

        remove_file : `bool`
            If True, it removes the `filtered` file. This variable is set
            to `False` by default.

        return_pd : `bool`
            If `True`, it returns the DataFrame containing the `filtered`
            version of `catl_kind`. This variable is set to `True` by
            default.

        Returns
        ------------
        catl_filt_data : `pd.DataFrame`
            DataFrame containing the `filtered` data for a given `catl_kind`
            file. This variale is **ONLY** returned if `return_pd == True`.
        """
        ## Checking input parameters
        #
        # `catl_kind`
        if not (catl_kind in ['master']):
            msg = '`catl_kind` ({0}) is not a valid input variable!'
            msg = msg.format(catl_kind)
            raise ValueError(msg)
        #
        # `remove_file`
        if not (isinstance(remove_file, bool)):
            msg = '`remove_file` ({0}) is not a valid input type!'
            msg = msg.format(type(remove_file))
            raise TypeError(msg)
        #
        # `return_pd`
        if not (isinstance(return_pd, bool)):
            msg = '`return_pd` ({0}) is not a valid input type!'
            msg = msg.format(type(return_pd))
            raise TypeError(msg)
        ## `Filtered` filepath
        filtered_filepath = self.catl_filtered_filepath(catl_kind=catl_kind,
                                check_exist=False, ext=ext)
        # Checking if file exists
        if os.path.exists(filtered_filepath):
            if remove_file:
                os.remove(filtered_filepath)
                create_file_opt = True
            else:
                create_file_opt = False
        else:
            create_file_opt = False
        ## Running only if needed
        if create_file_opt:
            ## Extracting data from `raw` catalogue
            catl_raw_pd = self.extract_input_catl_data(catl_kind=catl_kind)
            ## Filtered data
            catl_filt_pd = catl_raw_pd.loc[
                            (catl_raw_pd[self.mband_1] > self.mag_max) &
                            (catl_raw_pd[self.mband_1] < self.mag_min) &
                            (catl_raw_pd[self.mband_2] > self.mag_max) &
                            (catl_raw_pd[self.mband_2] < self.mag_min) &
                            (num.abs(catl_raw_pd[self.mband_1] - catl_raw_pd[self.mband_2]) <= self.mag_diff_tresh)]
            ##
            ## Resetting indices
            catl_filt_pd.reset_index(drop=True, inplace=True)
            ## Saving to disk
            catl_filt_pd.to_hdf(filtered_filepath, key='clusters', mode='w')
            cfutils.File_Exists(filtered_filepath)
        else:
            catl_filt_pd = pd.read_hdf(filtered_filepath)

        if return_pd:
            return catl_filt_pd

























