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
        # Assigning variables
        self.var1 = kwargs.get('var1', 0)