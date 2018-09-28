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

import numpy as np
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

from src.redseq_tools import RedSeq

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
                        choices=['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I',
                        'MAG_AUTO_Z','MAG_AUTO_Y'],
                        default='MAG_AUTO_G')
    ## 2nd Magnitude band
    parser.add_argument('-mband_2',
                        dest='mband_2',
                        help='Second apparent magnitude band to analyze.',
                        type=str,
                        choices=['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I',
                        'MAG_AUTO_Z','MAG_AUTO_Y'],
                        default='MAG_AUTO_Z')
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
                        default=24.)
    ## Upper magnitude limit for `mband_1` and `mband_2.`
    parser.add_argument('-mag_max',
                        dest='mag_max',
                        help="""
                        Upper magnitude limit for `mband_1` and `mband_2`.
                        """,
                        type=float,
                        default=17.)
    ## Aperture radius in 'arcseconds'
    parser.add_argument('-radius_size',
                        dest='radius_size',
                        help='Size of radius on the Sky. In units of `arcsec`',
                        type=_check_pos_val,
                        default=5.)
    ## Cosmology Choice
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Choice of Cosmology',
                        type=str,
                        choices=['WMAP7'],
                        default='WMAP7')
    ## Redshift bin size
    parser.add_argument('-z_binsize',
                        dest='z_binsize',
                        help='Size of bin for redsift `z`',
                        type=_check_pos_val,
                        default=0.0125)
    ## Minimum redshift value
    parser.add_argument('-z_min',
                        dest='z_min',
                        help='Minimim redshift to analyze.',
                        type=_check_pos_val,
                        default=0.4)
    ## Minimum redshift value
    parser.add_argument('-z_max',
                        dest='z_max',
                        help='Maximum redshift to analyze.',
                        type=_check_pos_val,
                        default=1.0)
    ## Choice of the input galaxy cluster location
    parser.add_argument('-input_catl_loc',
                        dest='input_catl_loc',
                        help='Choice of the input galaxy cluster location.',
                        type=str,
                        choices=['RedMapper', 'SDSS'],
                        default='RedMapper')
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
    Prog_msg = param_dict['Prog_msg']
    ## Making sure that `mag_diff_tresh` larger than some value
    if not (param_dict['mag_diff_tresh'] <= 4.):
        msg = '{0} `mag_diff_tresh` ({1}) must be smaller than 4! '
        msg += 'Exiting!'
        msg = msg.format(Prog_msg, param_dict['mag_diff_tresh'])
        raise ValueError(msg)
    ## Checking that `mag_min` is smaller than `mag_max`
    if not (param_dict['mag_min'] > param_dict['mag_max']):
        msg = '{0} `mag_min` ({1}) must be larger than `mag_max` ({2})! '
        msg += 'Exiting!'
        msg = msg.format(Prog_msg, param_dict['mag_min'],
            param_dict['mag_max'])
        raise ValueError(msg)
    ## Checking that `z_min` is larger than 0.4
    if not (param_dict['z_min'] >= 0.4):
        msg = '{0} `z_min` ({1}) must be larger than `0.4`! '
        msg += 'Exiting!'
        msg = msg.format(Prog_msg, param_dict['z_min'])
        raise ValueError(msg)

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
    ## Path to the output directory of the 'analysis' step.
    analysis_outdir = param_dict['rs_args'].analysis_out_dir(check_exist=False,
        create_dir=True)
    ## Adding to dictionary
    proj_dict['analysis_outdir'] = analysis_outdir
    
    return proj_dict

## ---------------------------- Main Analysis --------------------------------#

## Slicing clusters
def slice_clusters_idx(rm_pd, z_bins):
    """
    Returns all clusters from the DataFrame of clusters between
    a range of specified redshift limits.

    Parameters
    -----------
    rm_pd : `pd.DataFrame`
        DataFrame of cluster centers with ['RA','DEC','Z_LAMBDA'] information.

    z_bins : `numpy.ndarray`
        Array of the left and right bin edges of the different redshift bins.

    Returns
    -----------
    cluster_idx_arr : `list`
        List of indices of the cluster centers for each of the different
        redshift bins.
    """
    # Array of cluster indices
    cluster_idx_arr = [[] for x in range(len(z_bins))]
    # Populating array with cluser indices
    for kk, (zz_low, zz_high) in enumerate(z_bins):
        # Cluster indices
        cluster_idx_kk = rm_pd.loc[ (rm_pd['Z_LAMBDA'] >= zz_low) &
                                    (rm_pd['Z_LAMBDA'] < zz_high)].index.values
        # Saving to list
        cluster_idx_arr[kk] = cluster_idx_kk

    return cluster_idx_arr


def analysis_main(param_dict, proj_dict):
    """
    Main analysis of the Red Sequence. It produces the necessary files of
    the Red Sequence at given redshifts, magnitudes, etc.

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    # Constants
    Prog_msg = param_dict['Prog_msg']
    ## Reading in data
    # `Master` catalogue
    master_pd = param_dict['rs_args'].extract_filtered_data(catl_kind='master')
    # `Random` catalogue
    rand_pd = param_dict['rs_args'].extract_input_catl_data(catl_kind='random')
    # `RedMapper`/Cluster catalogue
    rm_pd = param_dict['rs_args'].extract_input_catl_data(catl_kind='redmapper')
    ##
    ## Range of redshifts to use
    z_arr = np.arange(  param_dict['z_min'],
                        param_dict['z_max'],
                        param_dict['z_binsize'])
    # Tuples of redshift bins
    z_bins = np.array([[z_arr[kk], z_arr[kk+1]] for kk in range(len(z_arr)-1)])
    #
    # Cluster indices for `rm_pd`
    cluster_idx_arr = slice_clusters_idx(rm_pd, z_bins)





def main(args):
    """
    Function analyze the `Red Sequence` using by cross-matching catalogues
    from `RedMapper` (http://risa.stanford.edu/redmapper/) and NOAO Data-Lab
    catalogue.
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    # Creating instance of `ReadML` with the input parameters
    param_dict['rs_args'] = RedSeq(**param_dict)
    ## Adding extra variables
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    proj_dict = param_dict['rs_args'].proj_dict
    proj_dict = directory_skeleton(param_dict, proj_dict)
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ## -- Main Analysis -- ##

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
