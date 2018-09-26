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
import numpy as num

from cosmo_utils.utils import file_utils as cfutils
from cosmo_utils.utils import work_paths as cwpaths
from cosmo_utils.utils import web_utils  as cweb

from redseq_tools import RedSeq

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import subprocess

# NOAO Data Client
from dl import authClient as ac
from dl import queryClient as qc
from getpass import getpass

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
                        help='First absolute magnitude band to analyze.',
                        type=str,
                        choices=['g','r','i','z','Y'],
                        default='g')
    ## 2nd Magnitude band
    parser.add_argument('-mband_2',
                        dest='mband_2',
                        help='Second absolute magnitude band to analyze.',
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
    if not (param_dict['mag_diff_tresh'] > 4.):
        msg = '{0} `mag_diff_tresh` ({1}) must be larger than 4! '
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

    return param_dict

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
    SQL_QUERY="""

    """
    ## Paths to online files
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
    param_dict['url_catl'         ] = url_catl
    param_dict['download_catl_opt'] = download_catl_opt
    param_dict['rand_url'         ] = rand_url
    param_dict['redmap_url'       ] = redmap_url

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
    master_dir_path = param_dict['rs_args'].input_catl_file_path(
        catl_kind='master', check_exist=False, create_dir=True)
    ## Random catalogue
    rand_dir_path = param_dict['rs_args'].input_catl_file_path(
        catl_kind='random', check_exist=False, create_dir=True)
    ## RedMapper catalogue
    redmap_dir_path = param_dict['rs_args'].input_catl_file_path(
        catl_kind='redmapper', check_exist=False, create_dir=True)
    ##
    ## Saving to dictionary `proj_dict`
    proj_dict['master_dir_path'] = master_dir_path
    proj_dict['rand_dir_path'] = rand_dir_path
    proj_dict['redmap_dir_path'] = redmap_dir_path

    return proj_dict

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
    ###
    ## Creating command to execute download
    for catl_kind in ['data', 'mocks']:
        ## Downloading directories from the web
        # Data
        if catl_kind == 'data':
            ## Prefix for the main directory
            catl_kind_prefix = os.path.join(param_dict['url_catl'],
                                            catl_kind,
                                            param_dict['catl_type'],
                                            'Mr'+param_dict['sample_s'])
            # Number of directories to cut/skip
            # See `wget` documentation for more details.
            cut_dirs = 8
        # Mocks
        if catl_kind == 'mocks':
            catl_kind_prefix = os.path.join(
                param_dict['url_catl'],
                catl_kind,
                'halos_{0}'.format(param_dict['halotype']),
                'dv_{0}'.format(param_dict['dv']),
                'hod_model_{0}'.format(param_dict['hod_n']),
                'clf_seed_{0}'.format(param_dict['clf_seed']),
                'clf_method_{0}'.format(param_dict['clf_method']),
                param_dict['catl_type'],
                param_dict['sample_Mr'])
            # Number of directories to cut/skip
            # See `wget` documentation for more details.
            cut_dirs = 13
        ##
        ## Direcotories for `members` and `groups`
        catl_kind_memb  = os.path.join(  catl_kind_prefix,
                                        'member_galaxy_catalogues/')
        catl_kind_group = os.path.join(  catl_kind_prefix,
                                        'group_galaxy_catalogues/')
        ## Checking if URL exists
        cweb.url_checker(catl_kind_memb)
        cweb.url_checker(catl_kind_group)
        ## String to be executed
        if param_dict['verbose']:
            cmd_dw = 'wget -m -nH -x -np -r -c --accept=*.hdf5 --cut-dirs={0} '
            cmd_dw += '--reject="index.html*" {1}'
        else:
            cmd_dw = 'wget -m -nH -x -np -r -c -nv --accept=*.hdf5 '
            cmd_dw += '--cut-dirs={0} --reject="index.html*" {1}'
        cmd_dw_m = cmd_dw.format(cut_dirs, catl_kind_memb)
        cmd_dw_g = cmd_dw.format(cut_dirs, catl_kind_group)
        ## Executing command
        print('{0} Downloading Dataset......'.format(param_dict['Prog_msg']))
        # Members
        print(cmd_dw_m)
        subprocess.call(cmd_dw_m, shell=True, cwd=proj_dict[catl_kind+'_out_m'])
        # Groups
        print(cmd_dw_g)
        subprocess.call(cmd_dw_g, shell=True, cwd=proj_dict[catl_kind+'_out_g'])
        ## Deleting `robots.txt`
        # Members
        try:
            os.remove('{0}/robots.txt'.format(proj_dict[catl_kind+'_out_m']))
        except:
            pass
        # Groups
        try:
            os.remove('{0}/robots.txt'.format(proj_dict[catl_kind+'_out_g']))
        except:
            pass
        ##
        ##
        print('\n\n{0} Catalogues were saved at: {1} and {2}\n\n'.format(
            param_dict['Prog_msg'], proj_dict[catl_kind+'_out_m'],
            proj_dict[catl_kind+'_out_g']))
        ##
        ## --- Perfect Catalogue -- Mocks
        if (catl_kind == 'mocks') and (param_dict['perf_opt']):
            ## Downloading directories from the web
            catl_kind_prefix = os.path.join(param_dict['url_catl'],
                                        catl_kind,
                                        param_dict['catl_type'],
                                        'Mr'+param_dict['sample_s'])
            ##
            ## Direcotories for `members` and `groups`
            catl_kind_memb  = os.path.join(  catl_kind_prefix,
                                            'perfect_member_galaxy_catalogues/')
            catl_kind_group = os.path.join(  catl_kind_prefix,
                                            'perfect_group_galaxy_catalogues/')
            ## Checking URLs
            cweb.url_checker(catl_kind_memb)
            cweb.url_checker(catl_kind_group)
            ## String to be executed
            cmd_dw = 'wget -r -nH -x -np -A *Mr{0}*.hdf5 --cut-dirs={1} '
            cmd_dw += '-R "index.html*" {2}'
            # Members and Groups commands
            cmd_dw_m = cmd_dw.format(param_dict['sample_s'],
                cut_dirs, catl_kind_memb)
            cmd_dw_g = cmd_dw.format(param_dict['sample_s'],
                cut_dirs, catl_kind_group)
            ## Executing command
            print('{0} Downloading Dataset......'.format(param_dict['Prog_msg']))
            print(cmd_dw_m)
            subprocess.call(cmd_dw_m, shell=True, 
                cwd=proj_dict['mocks_out_perf_memb'])
            print(cmd_dw_g)
            subprocess.call(cmd_dw_g, shell=True, 
                cwd=proj_dict['mocks_out_perf_groups'])

            ## Deleting `robots.txt`
            os.remove('{0}/robots.txt'.format(proj_dict['mocks_out_perf_memb']))
            ##
            ##
            print('\n\n{0} Catalogues were saved at: {1}\n\n'.format(
                param_dict['Prog_msg'], proj_dict['mocks_out_perf_memb']))

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



    ##
    ## Checking if there is a local version of the catalogues
    if param_dict['download_catl_opt']:
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
        ##
        ## Downloading necessary data
        download_directory(param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
