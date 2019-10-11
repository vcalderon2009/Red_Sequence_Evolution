from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import time
import easyaccess as ea
import pandas as pd
from scipy import stats
import cPickle
import string
import sys
# from astropy.table import Table
# from plot_utils_mod import plot_2d_dist
# from plot_utils_mod import plot_pretty
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
import numpy.random as rnd
from scipy.ndimage import gaussian_filter

# plot_pretty()

if __name__ == '__main__':

    load = True
    if load:
        f = open('query_mi_rz.pkl', 'rb')
        MASTER_DATA = cPickle.load(f)
        f.close()
        t0 = time.time()
    else:
        # Connecting to the DES DB
        connection = ea.connect()
        print 'connection successful'

        query = 'select MAG_AUTO_R, MAG_AUTO_Z, RA, DEC, MAG_AUTO_I from\
         SVA1_GOLD where MAG_AUTO_R > 17 AND MAG_AUTO_R < 24 AND\
          MAG_AUTO_Z < 25 AND ABS(MAG_AUTO_Z-MAG_AUTO_R) < 4'
        t0 = time.time()
        MASTER_DATA = connection.query_to_pandas(query)  # one line!

        f = open('query_mi_rz.pkl', 'wb')
        cPickle.dump(MASTER_DATA, f)
        f.close()
    
    t1 = time.time()
    

def radius_cosmo(z):
    """
    Returns the degrees per Megaparsec value as a function of redshift
    (dependent on cosmology)

    Parameters:
    ------------
    z : `float`
        Input redshift

    Returns:
    ------------ 
    deg_per_mpc : `float`
        Float value of degrees per megaparsec, depending on cosmology
    """
    deg_per_mpc = (1000./60. * (1. / cosmo.kpc_proper_per_arcmin(z))).value

    return deg_per_mpc


def slice_clusters(cluster_data, z, dz):
    """
    Returns all clusters from cluster_data between redshifts z-dz and z+dz

    Parameters:
    ------------
    cluster_data : `numpy.ndarray`
        Input cluster catalogue

    z : `float`
        Input redshift, the center of the redshift bin

    dz : `float`
        Input redshift increment/decrement

    Returns:
    ------------
    clusters : `numpy.ndarray`
        All clusters within z-dz and z+dz
    """
    clusters = cluster_data[(cluster_data['z_lambda'] >= (z - (dz / 2.)))
                            & (cluster_data['z_lambda'] <= (z + (dz / 2.)))]
    
    return clusters

#============================================
# def slice_random(random_data, z, dz):
#     randoms = random_data[(random_data['Z'] >= (z - (dz / 2)))
#                           & (random_data['Z'] <= (z + (dz / 2)))]
#     return randoms
#============================================


def get_mags(data, redshift, MASTER_DATA=MASTER_DATA):
    """
    Returns magnitudes (corresponding to a 2D histogram) from background 
    regions based on the clusters catalogue

    Parameters:
    ------------
    data : `numpy.ndarray`
        Input clusters catalogue, containing ra,dec information for
        galaxy clusters

    redshift : `float`
        Input redshift, the center of the redshift bin

    MASTER_DATA : `numpy.ndarray`
        master data catalogue from input photometric survey, containing 
        ra,dec, mags information. This is MASTER_DATA i.e. the largest
        photometry table, by default
    
    Returns:
    ------------
    r_array, z_array, i_array : `numpy.ndarray`
        arrays that contain magnitude information for r,z,i bands
    """
    radius = radius_cosmo(redshift)

    r_array = []
    z_array = []
    i_array = []
    for j, d in enumerate(data):
        ra = d['RA']
        dec = d['DEC']
        r = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_R']
        z = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_Z']
        i = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_I']
        r_array.extend(r)
        z_array.extend(z)
        i_array.extend(i)

        print '%i of %i' % (j, len(data))

    r_array = np.asarray(r_array)
    z_array = np.asarray(z_array)
    i_array = np.asarray(i_array)

    return r_array, z_array, i_array


def get_counts(data, redshift, dz, xbins=xbins, ybins=ybins):
    """
    Returns counts (corresponding to a 2D histogram) from cluster regions 
    based on the cluster catalogues

    Parameters:
    ------------
    random_data : `numpy.ndarray`
        Input cluster catalogue in redshift slices, containing ra,dec 
        information

    redshift : `float`
        Input redshift, the center of the redshift bin

    dz : `float`
        Input redshift increment, half of the width of the redshift bin

    xbins : `int`
        bins along x-axis for magnitude

    ybins : `int`
        bins along y-axis for color
    
    Returns:
    ------------
    counts : `numpy.ndarray`
        2D array that maps the cluster counts in a particular redshift slice 
        to array elements based on clusterss catalogue
    """
    num_clusters = len(data)
    r, z, i = get_mags(data, redshift)
    rz = r - z
    counts, _, _ = np.histogram2d(i, rz, bins=[xbins, ybins])

    """
    Dividing total counts by pi * total number of clusters
    """
    counts /= (np.pi * num_clusters)

    return counts

def get_background(random_data, xbins, ybins, z, MASTER_DATA = MASTER_DATA):
    """
    Returns counts (corresponding to a 2D histogram) from background regions 
    based on the randoms catalogue

    Parameters:
    ------------
    random_data : `numpy.ndarray`
        Input randoms catalogue, containing ra,dec information

    xbins : `int`
        bins along x-axis for magnitude

    ybins : `int`
        bins along y-axis for color

    z : `float`
        Input redshift, the center of the redshift bin

    MASTER_DATA : `numpy.ndarray`
        master data catalogue from input photometric survey, containing ra,dec,
        mags information
    
    Returns:
    ------------
    counts : `numpy.ndarray`
        2D array that maps the background counts to array elements based on 
        randoms catalogue
    """

    radius = radius_cosmo(z)
    
    r_array = []
    z_array = []
    i_array = []
    for data in random_data:
        ra = data['RA']
        dec = data['DEC']
        r = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_R']
        z = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_Z']
        i = MASTER_DATA[np.sqrt((MASTER_DATA['RA'] - ra)**2 + \
            (MASTER_DATA['DEC'] - dec)**2) < radius]['MAG_AUTO_I']
        r_array.extend(r)
        z_array.extend(z)
        i_array.extend(i)
    r = np.asarray(r_array)
    z = np.asarray(z_array)
    i = np.asarray(i_array)
    rz = r-z
    counts, _, _ = np.histogram2d(i, rz, bins=[xbins, ybins])
    """
    Dividing total counts by pi * total number of clusters
    """
    counts /= (np.pi * len(random_data))

    return counts


# Stuff already defined == xedges_random,yedges_random,counts_random_avg
def func_plot(counts, counts_random, xbins, ybins, z, dz,sigma=1.25):
    """
    Generates a plot (corresponding to a 2D histogram) by subtracting the 
    clusters 2D histogram and the randoms 2D histogram, and smoothing the 
    result with a gaussian filter

    Parameters:
    ------------
    counts : `numpy.ndarray`
        2D array that maps the background counts to array elements based on 
        clusters catalogue

    counts_random : `numpy.ndarray`
        2D array that maps the background counts to array elements based on 
        randoms catalogue

    xbins : `int`
        bins along x-axis for magnitude

    ybins : `int`
        bins along y-axis for color

    z : `float`
        Input redshift, the center of the redshift bin

    dz : `float`
        Input redshift increment, half of the width of the redshift bin

    sigma : `float`
        Width of the gaussian smoothing of the subtracted value of 
        cluster and random counts per pixel
    
    Returns:
    ------------
    A 2D matplotlib plot
    """
    smooth = gaussian_filter(counts.T - counts_random.T,sigma=sigma)
    plt.figure(figsize=(7, 4))
    plt.imshow(smooth, origin='lower', extent=[
               xbins.min(), xbins.max(), ybins.min(), ybins.max()], \
               aspect='auto', cmap='viridis',vmin=0,interpolation='nearest')
    plt.xlim(17, 24)
    plt.ylim(-1, 3)
    plt.xlabel(r'$m_{\rm i}$')
    plt.ylabel(r'$r-z$')
    plt.title(r'$\mathrm{z = %.3f, \Delta z = %.3f}$' %(z, dz))
    # plt.colorbar()
    #plt.savefig('test/output_cmd_sva1gold_2/cmd_clusters_zbin_%.3f_%.3f.png'\
    #%(z, dz), bbox_inches='tight')


# MAIN - HERE WE GO!
def generate_cmd(redshifts, dz, cluster_data, random_data, xbins, ybins):
    """
    Takes redshift bins, cluster and randoms data as input, and generates a 
    count-in-cell analysis plot (corresponding to a 2D histogram) as the output

    Parameters:
    ------------
    redshifts : `float`
        Input redshift, the center of the redshift bin

    dz : `float`
        Input redshift increment, half of the width of the redshift bin

    cluster_data : `numpy.ndarray`
        Input clusters catalogue, containing ra,dec information

    random_data : `numpy.ndarray`
        Input randoms catalogue, containing ra,dec information

    xbins : `int`
        bins along x-axis for magnitude

    ybins : `int`
        bins along y-axis for color
    
    Returns:
    ------------
    Output plots for this project - a count-in-cell 2D histogram demonstrating
    the smoothed features in the desired parameter space
    """
    for z in redshifts:
        print 'z = %.3f' % z
        print
        cluster_slice = slice_clusters(cluster_data, z, dz)
        counts_cluster = get_counts(cluster_slice, z, dz)
        counts_random = get_background(random_data, xbins, ybins, z)
        func_plot(counts_cluster, counts_random, xbins, ybins, z, dz)

#=======================================

if __name__ == "__main__":
    xbins = np.linspace(17, 24, 200)
    ybins = np.linspace(-4, 4, 200)

    """
    Download cluster location catalogue from Redmapper?
    """
    hdulist_cluster = fits.open('redmapper_sva1_public_v6.3_catalog.fits')
    cluster_data = hdulist_cluster[1].data
    hdulist_cluster.close()

    print 'prelim redmapper info generated'

    """
    Download 'randoms' catalogue from Redmapper?
    """
    hdulist_random = fits.open('redmapper_sva1_public_v6.3_randoms.fits')
    random_data = hdulist_random[1].data
    random_data = rnd.choice(random_data, 75)
    hdulist_random.close()
    
    print 'prelim randoms info generated'

    """
    Decide width of redshift slice
    """
    dz = 0.0125

    """
    Array of redshifts with specific redshift slice
    """
    redshifts = np.arange(0.45,1.0,dz)


    generate_cmd(redshifts, dz, cluster_data, random_data, xbins, ybins)

    #####
    t2 = time.time()
    print t1-t0, t2-t1