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
        self.mband_1        = kwargs.get('mband_1', 'g')
        self.mband_2        = kwargs.get('mband_2', 'z')
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

    def input_catl_prefix_str(self, catl_kind='master', ext='csv'):
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
        input_catl_dir = os.path.join(   self.proj_dict['ext_dir'],
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
            if not (os.path.exists(input_catl_dir)):
                msg = '`input_catl_dir` ({0}) was not found!'.format(
                    input_catl_dir)
                raise FileNotFoundError(msg)

        return filepath





















