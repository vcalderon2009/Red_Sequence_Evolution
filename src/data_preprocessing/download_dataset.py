#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-09-26
# Last Modified: 2018-09-26
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Downloads the necessary catalogue from the web to perform the
analysis on the `Red Sequence` project.
"""
# Importing Modules
import os
import sys
import numpy as np

from cosmo_utils.utils import file_readers as cfreaders
from cosmo_utils.utils import file_utils   as cfutils
from cosmo_utils.utils import work_paths   as cwpaths
from cosmo_utils.utils import web_utils    as cweb

from src.redseq_tools import RedSeq

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import subprocess

# Tap Service
from tap import TAP_Service

# NOAO Data Client

### ----| Common Functions |--- ###

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
    val: int or float
        value to be evaluated by `val_min`

    val_min: float or int, optional (default = 0)
        minimum value that `val` can be

    Returns
    -------
    ival: float
        value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError: Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Downloads the necessary catalogues from the web'
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
    ##
    Prog_msg = param_dict['Prog_msg']
    ## Testing if `wget` exists in the system
    if is_tool('wget'):
        pass
    else:
        msg = '{0} You need to have `wget` installed in your system to run '
        msg += 'this script. You can download the entire dataset at {1}.\n\t\t'
        msg += 'Exiting....'
        msg = msg.format(Prog_msg, param_dict['url_catl'])
        raise ValueError(msg)
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
            param_dict['param_max'])
        raise ValueError(msg)
    ## Check that no magnitude is the same
    if (np.unique([param_dict['mband_1'], param_dict['mband_2'], param_dict['mband_3']]).size != 3):
        msg = '{0} All three magnitude bands must be different: `{1}`, `{2}`, '
        msg += '`{4}`! Exiting!'
        msg = msg.format(   param_dict['Prog_msg'], param_dict['mband_1'],
                            param_dict['mband_2'], param_dict['mband_3'])
        raise ValueError(msg)

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    Returns
    ----------
    param_dict: python dictionary
        dictionary with old and new values added
    """
    ##
    ## Creating SQL Query for cleaning the data
    sql_query = 'SELECT TOP {0} RA, DEC, {1}, {2}, {3} FROM des_dr1.galaxies'
    sql_query = sql_query.format(   param_dict['master_limit'],
                                    param_dict['mband_1'],
                                    param_dict['mband_2'],
                                    param_dict['mband_3'])
    ## Paths to online files
    # Master catalogue
    # master_url = os.path.join(  'http://lss.phy.vanderbilt.edu/groups',
    #                             'Red_Sequence',
    #                             'test.csv')
    # cweb.url_checker(master_url)
    # Random catalogue
    rand_url = os.path.join('http://risa.stanford.edu/redmapper/v6.3',
                            'redmapper_sva1_public_v6.3_randoms.fits.gz')
    cweb.url_checker(rand_url)
    # RedMapper catalouge
    redmap_url = os.path.join('http://risa.stanford.edu/redmapper/v6.3',
                            'redmapper_sva1_public_v6.3_catalog.fits.gz')
    cweb.url_checker(redmap_url)
    ###
    ### To dictionary
    # param_dict['master_url'] = master_url
    param_dict['rand_url'  ] = rand_url
    param_dict['redmap_url'] = redmap_url
    param_dict['sql_query' ] = sql_query

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
    ## Master catalogue
    master_dir_path = param_dict['rs_args'].input_catl_dir_path(
        catl_kind='master', check_exist=False, create_dir=True)
    ## Random catalogue
    rand_dir_path = param_dict['rs_args'].input_catl_dir_path(
        catl_kind='random', check_exist=False, create_dir=True)
    ## RedMapper catalogue
    redmap_dir_path = param_dict['rs_args'].input_catl_dir_path(
        catl_kind='redmapper', check_exist=False, create_dir=True)
    ##
    ## Saving to dictionary `proj_dict`
    proj_dict['master_dir_path'] = master_dir_path
    proj_dict['rand_dir_path'  ] = rand_dir_path
    proj_dict['redmap_dir_path'] = redmap_dir_path

    return proj_dict

### ----| Downloading Data |--- ###

class DataLab(TAP_Service):
    def __init__(self, *args, **kwargs):
        host = 'datalab.noao.edu'
        path = '/tap'
        port = 80
        kwargs['protocol'] = "https"
        TAP_Service.__init__(self, host, path, port, *args, **kwargs)

### ----| Downloading Data |--- ###

def download_directory(param_dict, proj_dict):
    """
    Downloads the necessary catalogues to perform the analysis

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    ## Creating command to execute download
    kind_arr = ['random'       , 'redmapper'      ]
    keys_arr = ['rand_dir_path', 'redmap_dir_path']
    url_arr  = ['rand_url'     , 'redmap_url'     ]
    # Looping over each instance
    for kk, (kind_kk, key_kk, url_kk) in enumerate(zip(kind_arr, keys_arr, url_arr)):
        if param_dict['verbose']:
            print('{0} Downloading: `{1}`'.format(param_dict['Prog_msg'],
                kind_kk))
        ## Paths to local files
        kk_local = param_dict['rs_args'].input_catl_file(catl_kind=kind_kk,
                        check_exist=False)
        kk_remote = param_dict[url_kk]
        ## Deleting if necessary
        if (os.path.exists(kk_local)):
            # Checking if to delete the local copy of the file or not.
            if param_dict['remove_files']:
                # Removing file
                os.remove(kk_local)
                ## Downloading file
                cfutils.File_Download_needed(kk_local, kk_remote)
            ##
            msg = '{0} Local copy can be found at: {1}'.format(
                param_dict['Prog_msg'], kk_local)
            print(msg)
        else:
            ##
            msg = '{0} Local copy can be found at: {1}'.format(
                param_dict['Prog_msg'], kk_local)
            print(msg)
    ##
    ## Downloading MASTER catalogue if necessary
    master_local = param_dict['rs_args'].input_catl_file(catl_kind='master',
                        check_exist=False)
    # Checking if file exists
    if os.path.exists(master_local):
        if param_dict['remove_files']:
            # Removing file
            os.remove(master_local)
            # Option for creating new MASTER catalogue
            master_calc = True
        else:
            master_calc = False
    else:
        master_calc = True
    ## Computing creating new catalogue if necessary
    if master_calc:
        ## Extracting catalogue
        print('{0} Downloading Master catalogue using SQL Query: '.format(
                    param_dict['Prog_msg']))
        print('{0} >> SQL Query: {1}'.format(param_dict['Prog_msg'],
            param_dict['sql_query']))
        # Downloading dataset
        DL          = DataLab()
        results_tab = DL.query(param_dict['sql_query'])
        results_pd  = results_tab.to_pandas()
        # Saving to Master file
        cfreaders.pandas_df_to_hdf5_file(results_pd, master_local, '/gals')
        cfutils.File_Exists(master_local)
    # Master File
    print('{0} Master catalogue can be found at: {1}'.format(
                param_dict['Prog_msg'], master_local))
    # Message
    if param_dict['verbose']:
        print('{0} Download complete!'.format(param_dict['Prog_msg']))

### ----| Main Function |--- ###

def main(args):
    """
    Downloads the necessary catalogues to perform the 1- and 2-halo 
    conformity analysis
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Checking for correct input
    param_vals_test(param_dict)
    # Creating instance of `ReadML` with the input parameters
    param_dict['rs_args'] = RedSeq(**param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Creating folder directory
    proj_dict = param_dict['rs_args'].proj_dict
    proj_dict = directory_skeleton(param_dict, proj_dict)
    ## Downloading data
    download_directory(param_dict, proj_dict)
    # Cleaning up the data and saving to file
    param_dict['rs_args'].extract_filtered_data(catl_kind='master',
        return_pd=False, remove_file=param_dict['remove_files'])

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
