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

# Astropy-related
import astropy.cosmology   as astrocosmo
import astropy.constants   as ac
import astropy.units       as u
import astropy.table       as astro_table
import astropy.coordinates as astrocoord

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
                        choices=['mag_auto_g', 'mag_auto_r', 'mag_auto_i',
                        'mag_auto_z', 'mag_auto_y'],
                        default='mag_auto_g')
    ## 2nd Magnitude band
    parser.add_argument('-mband_2',
                        dest='mband_2',
                        help='Second apparent magnitude band to analyze.',
                        type=str,
                        choices=['mag_auto_g', 'mag_auto_r', 'mag_auto_i',
                        'mag_auto_z', 'mag_auto_y'],
                        default='mag_auto_z')
    ## 3rd Magnitude band
    parser.add_argument('-mband_3',
                        dest='mband_3',
                        help='Third apparent magnitude band to analyze.',
                        type=str,
                        choices=['mag_auto_g', 'mag_auto_r', 'mag_auto_i',
                        'mag_auto_z', 'mag_auto_y'],
                        default='mag_auto_i')
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
    ## Maximum number of elements to download
    parser.add_argument('-master_limit',
                        dest='master_limit',
                        help='Number of elements to use for the MASTER file',
                        type=int,
                        default=100000)
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
                        choices=['WMAP5','WMAP7','WMAP9','Planck15','custom'],
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
    ## Choice of binning
    parser.add_argument('-hist_nbins',
                        dest='hist_nbins',
                        help='Number of bins for x- and y-axis.',
                        type=_check_pos_val,
                        default=200)
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
    ## Check that no magnitude is the same
    if (np.unique([param_dict['mband_1'], param_dict['mband_2'], \
        param_dict['mband_3']]).size != 3):
        msg = '{0} All three magnitude bands must be different: `{1}`, `{2}`, '
        msg += '`{4}`! Exiting!'
        msg = msg.format(   param_dict['Prog_msg'], param_dict['mband_1'],
                            param_dict['mband_2'], param_dict['mband_3'])
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
    ##
    ## Bin array along x- (magnitude )and y-axis (color)
    x_bins = np.linspace(   param_dict['mag_max'],
                            param_dict['mag_min'],
                            param_dict['hist_nbins'])
    y_bins = np.linspace(   -param_dict['mag_diff_tresh'],
                            param_dict['mag_diff_tresh'],
                            param_dict['hist_nbins'])
    ##
    ## Range of redshifts to use
    z_arr = np.arange(  param_dict['z_min'],
                        param_dict['z_max'],
                        param_dict['z_binsize'])
    # Redshift bins
    z_bins = np.array([[z_arr[kk], z_arr[kk+1]] for kk in range(len(z_arr)-1)])
    # Number of redshift bins
    n_z_bins = len(z_bins)
    # Centers fo the redshift bins
    z_centers = np.array([np.mean(xx) for xx in z_bins])
    # List of apparent magnitudes
    mbands_arr = [  param_dict['mband_1'],
                    param_dict['mband_2'],
                    param_dict['mband_3']]
    #
    # Saving to main dictionary
    param_dict['x_bins'    ] = x_bins
    param_dict['y_bins'    ] = y_bins
    param_dict['z_arr'     ] = z_arr
    param_dict['z_bins'    ] = z_bins
    param_dict['n_z_bins'  ] = n_z_bins
    param_dict['z_centers' ] = z_centers
    param_dict['mbands_arr'] = mbands_arr
    
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

## ------------------------- Tools and Functions -----------------------------#

# Choice of cosmological model
def cosmo_create(cosmo_choice='WMAP7', H0=100., Om0=0.25, Ob0=0.04,
    Tcmb0=2.7255):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    cosmo_choice : {'WMAP7', 'WMAP9', 'Planck', 'custom'} `str`
        The choice of cosmology for the project. This variable is set to
        'WMPA7' by default.

        Options:
            - WMAP5: Cosmology from WMAP-5 cosmology
            - WMAP7: Cosmology from WMAP-7 cosmology
            - WMAP9: Cosmology from WMAP-9 cosmology
            - Planck15: Cosmology from Plack-2015
            - custom: Custom cosmology. The user sets the default parameters.
    
    h: float, optional (default = 1.0)
        value for small cosmological 'h'.

    Returns
    ----------                  
    cosmo_model : `astropy.cosmology.core.FlatLambdaCDM`
        cosmology used throughout the project

    Notes
    ----------
    For more information regarding the choice of cosmology, the author is
    advised to read the docs: `http://docs.astropy.org/en/stable/cosmology/`
    """
    ## Choosing cosmology
    if (cosmo_choice == 'WMAP5'):
        cosmo_model = astrocosmo.WMAP5.clone(H0=H0)
    elif (cosmo_choice == 'WMAP7'):
        cosmo_model = astrocosmo.WMAP7.clone(H0=H0)
    elif (cosmo_choice == 'WMPA9'):
        cosmo_model = astrocosmo.WMAP9.clone(H0=H0)
    elif (cosmo_choice == 'Planck15'):
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif (cosmo_choice == 'custom'):
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ##
    ## Cosmological parameters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0

    return cosmo_model, cosmo_params

# Cosmological radius
def radius_cosmo(z, cosmo_model, units='deg_mpc'):
    """
    Returns the 'degrees per Mpc' or 'arcmin per kpc' value as a function of 
    redshift `z`.

    Parameters
    -----------
    z : `float`
        Input redshift.

    cosmo_model : `astropy.cosmology.core.FlatLambdaCDM`
        Cosmological model used in this analysis.

    units : {'deg_mpc', 'arcmin_kpc'}, `str`
        Choice of units for the radius. This variable is set to `deg_mpc`
        by default.

        Options:
            - 'deg_mpc': Degrees per Megaparsec
            - 'arcmin_kpc': Arcminutes per kiloparsec.

    Returns:
    ------------ 
    angle_dist_value : `float`
        Float value of degrees per megaparsec or arcminutes per kiloparsec.

    Notes
    ----------
    Returning angular distance value is dependent on redshift and the choice 
    of cosmology `cosmo_model`.
    """
    ## The distance in proper kpc corresponding to an arcmin at each `z`
    kpc_arcmin = cosmo_model.kpc_proper_per_arcmin(z)
    ## Transforming data
    if (units == 'arcmin_kpc'):
        angle_dist_value = (1. / kpc_arcmin).to(1.*u.arcmin / u.kpc).value
    elif (units == 'deg_mpc'):
        angle_dist_value = (1. / kpc_arcmin).to(1.*u.deg / u.Mpc).value

    return angle_dist_value

## Counts calculation for cluster regions
def cluster_counts(cluster_idx_arr, rm_pd, master_pd, param_dict):
    """
    Calculates the counts of a 2D-histogram for cluster regions based on
    the cluster catalogue `cluster_ii`

    Parameters
    -------------
    cluster_idx_arr : `numpy.ndarray`
        Set of indices for the different redshift bins `z_bins`

    rm_pd : `pd.DataFrame`
        Main DataFrame containing information about the cluster centers.

    master_pd : `pd.DataFrame`
        Master DataFrame from input photometry survey, containing
        [RA, DEC, MAG_AUTO_G, MAG_AUTO_Z] information. This is the 
        `Master Data`, i.e. the largest photometry table, by default.

    param_dict : `dict`
        Dictionary with `project` variables

    Returns
    -------------
    counts_z_arr : `numpy.ndarray`, shape (N,)
        Array containing total counts for the 2D-histograms at each redshisft
        bin in `z_bins`.
    """
    # Constants
    cols_used = ['RA', 'DEC']
    x_bins    = param_dict['x_bins']
    y_bins    = param_dict['y_bins']
    # Initializing `counts_z_arr` array
    counts_z_arr = [[] for x in range(param_dict['n_z_bins'])]
    # Radius at each redshift bin
    r_z_arr = np.array([radius_cosmo(xx, param_dict['cosmo_model']) \
                    for xx in param_dict['z_centers']])
    # Calculating counts at each redshift bin
    for kk in range(param_dict['n_z_bins']):
        # Indices for given redshift bin
        idx_kk       = cluster_idx_arr[kk]
        n_cluster_kk = len(idx_kk)
        # Checking if there's an empty bin
        if (n_cluster_kk > 0):
            # Set of RA and DEC values for that given redshift bin
            cluster_kk = rm_pd.loc[idx_kk, cols_used].reset_index(drop=True)
            # Extracting magnitudes for given cluster centers and member
            # galaxies
            mags_zz_arr = get_magnitudes(cluster_kk, r_z_arr[kk], master_pd,
                            param_dict)
            # Extracting magnitudes
            mband_1_kk, mband_2_kk, mband_3_kk = mags_zz_arr
            # Computing counts
            counts_kk, _, _ = np.histogram2d(   mband_3_kk,
                                                mband_1_kk - mband_2_kk,
                                                bins=[x_bins, y_bins])
            # Normalizing total counts
            counts_kk /= (np.pi * len(cluster_kk))
            # Saving counts
            counts_z_arr[kk] = counts_kk
        else:
            counts_z_arr[kk] = 0

    return counts_z_arr


## Extracting magnitudes for the various galaxy clusters.
def get_magnitudes(cluster_kk, r_kk_cen, master_pd, param_dict,
    dist_radius=10):
    """
    Extracts the magnitudes (corresponding to a 2D histogram) from
    background regions based on the cluters catalogue `cluster_kk`

    Parameters
    ------------
    cluster_kk : `pd.DataFrame`
        DataFrame containing coordinates of cluster centers.

    r_kk_cen : `float`
        Input redshift. The center of the redshift bin being analyzed.

    master_pd : `pd.DataFrame`
        DataFrame corresponding to the `master` data catalogue from input
        `photometric` survey. It contains the coordinates ('RA', 'DEC') and
        information about the three apparent magnitudes, `mband_1`,
        `mband_2`, and `mband_3`.

    dist_radius : `int`, optional
        Number of `unit_distance` away from the center. This value
        corresponds to the total number of ``kpc`` or ``Mpc`` away
        from the center. This variable is set to ``3`` by default.
    
    Returns
    ---------
    mags_zz_arr : list
        List containing the resulting apparent magnitudes of 
    """
    # Constants
    mbands_arr = param_dict['mbands_arr']
    # Temporary
    mbands_arr = [param_dict['mband_1'], param_dict['mband_2']]
    # Square of the radius `r_kk_cen`
    r_k_sq     = (dist_radius * r_kk_cen)**2
    # Initializing array
    mags_zz_arr = [[] for kk in range(len(cluster_kk))]
    # Looping over each coordinate
    tqdm_msg = 'Computing magnitudes: '
    for zz, (ra_zz, dec_zz) in enumerate(tqdm(cluster_kk.values, desc=tqdm_msg)):
        # 1st magnitude
        mags_zz_diff  = (master_pd['ra' ] - ra_zz)**2
        mags_zz_diff += (master_pd['dec'] - dec_zz)**2
        mags_zz_mask  = mags_zz_diff < r_k_sq
        # Only selecting those that match the criteria
        mags_zz_match = master_pd.loc[mags_zz_mask, mbands_arr]
        # Saving to array
        mags_zz_arr[zz] = mags_zz_match.values

    return mags_zz_arr

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
    #
    # Cluster indices for `rm_pd` (in redshift slice)
    cluster_idx_arr = slice_clusters_idx(rm_pd, param_dict['z_bins'])
    # Cluster counts
    counts_z_arr = cluster_counts(cluster_idx_arr, rm_pd, master_pd,
                        param_dict)
    # Background





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
    ## Choice of cosmology
    ## Choosing cosmological model
    (   cosmo_model ,
        cosmo_params) = cosmo_create(cosmo_choice=param_dict['cosmo_choice'])
    param_dict['cosmo_model' ] = cosmo_model
    param_dict['cosmo_params'] = cosmo_params
    ##
    ## Printing out project variables
    key_skip = ['Prog_msg', 'z_centers', 'z_bins', 'z_arr','y_bins','x_bins']
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if not (key in key_skip):
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ## -- Main Analysis -- ##
    analysis_main(param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
