#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-09-28
# Last Modified: 2018-09-28
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, Gourav Khullar, "]
__email__      = ['victor.calderon@vanderbilt.edu', 'gkhullar@uchicago.edu']
__maintainer__ = ['Victor Calderon']
"""
Main analysis for the 'Red Sequence' project.

Steps:
    1) Cross-match datasets and find common `ra, dec, mag1, and mag2`.
    2) Bin clusters in redshift-bins
    3) ...
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.mock_catalogues import catls_utils as cmcu

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)
from tqdm import tqdm

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## Functions
class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _check_pos_val(val, val_min=0):
    """
    Checks if value is larger than `val_min`

    Parameters
    ----------
    val : `int` or `float`
        Value to be evaluated by `val_min`

    val_min: `float` or `int`, optional
        minimum value that `val` can be. This value is set to `0` by default.

    Returns
    -------
    ival : `float`
        Value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError : Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Main analysis of the `Red Sequence` project.'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## 1st Magnitude band
    parser.add_argument('-mband_1',
                        dest='mband_1',
                        help='First apparent magnitude band to analyze.',
                        type=str,
                        choices=['g','r','i','z','Y'],
                        default='g')
    ## 2nd Magnitude band
    parser.add_argument('-mband_2',
                        dest='mband_2',
                        help='Second apparent magnitude band to analyze.',
                        type=str,
                        choices=['g','r','i','z','Y'],
                        default='z')
    ## Maximum difference between `mband_1` and `mband_2`
    parser.add_argument('-mag_diff_tresh',
                        dest='mag_diff_tresh',
                        help="""
                        Maximum threshold of the difference between `mband_1`
                        and `mband_2`. It must be larger than `4`.
                        """,
                        type=_check_pos_val,
                        default=4.)
    ## Bottom magnitude limit for `mband_1` and `mband_2.`
    parser.add_argument('-mag_min',
                        dest='mag_min',
                        help="""
                        Bottom magnitude limit for `mband_1` and `mband_2`.
                        """,
                        type=float,
                        default=24)
    ## Upper magnitude limit for `mband_1` and `mband_2.`
    parser.add_argument('-mag_max',
                        dest='mag_max',
                        help="""
                        Upper magnitude limit for `mband_1` and `mband_2`.
                        """,
                        type=float,
                        default=17)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help="""
                        Delete files from previous analyses with same
                        parameters
                        """,
                        type=_str2bool,
                        default=False)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cfutils.Program_Msg(__file__))
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with `project` variables

    Raises
    -----------
    ValueError : Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ##
    ## This is where the tests for `param_dict` input parameters go.

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict : `dict`
        dictionary with input parameters and values

    Returns
    ----------
    param_dict : `dict`
        dictionary with old and new values added
    """
    # This is where you define `extra` parameters for adding to `param_dict`.

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict : `dict`
        Dictionary with current and new paths to project directories
    """
    ## In here, you define the directories of your project

    return proj_dict

def main(args):
    """

    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## Adding extra variables
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)