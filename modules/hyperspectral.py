#!/usr/bin/env python

'''
    Module for various hyperspectral image routines.
'''

# System imports
import os
import sys
import pdb

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim

from scipy import interpolate
from scipy import signal
from numpy import genfromtxt
from skimage import color

import cv2

# Plotting
import matplotlib.pyplot as plt

from modules import utils
from modules import this_path

def implay(cube, delay=20):
    '''
        Play hyperspectral image as a video
    '''
    if cube.dtype != np.uint8:
        cube = (255*cube/cube.max()).astype(np.uint8)
    
    T = cube.shape[-1]
    
    for idx in range(T):
        cv2.imshow('Video', cube[..., idx])
        cv2.waitKey(delay)
    

def nb_filter(wvl, cwl, fwhm):
    '''
        Create a narrowband gaussian filter.

        Inputs:
            wvl: Wavelengths in nm
            cwl: Central wavelength in nm
            fwhm: Full width half max in nm

        Outputs:
            filt: Narrowband filter of the same dimensions as wvl.
    '''
    sigma = fwhm/(2*np.sqrt(2*np.log(2)));
    filt = np.exp(-pow((wvl - cwl), 2)/(2*sigma*sigma))

    return filt                  

def fwhm(wvl, spectrum):
    '''
        Compute Full-width Half Max for a given spectral signature.

        Inputs:
            wvl: Wavelengths
            spectrum: Spectral signature. WARNING: This function assumes that
                the spectrum has only one peak.

        Outputs:
            cwl: Center wavelength
            fwhm: FWHM around central wavelength
    '''
    # First find cwl as peak.
    cwl_idx = spectrum.argmax()

    # Use this CWL as input to peak_widths
    widths, width_heights, left_ips, right_ips = signal.peak_widths(spectrum,
                                                                    [cwl_idx])

    # Now find interpolated FWHM
    f = interpolate.interp1d(np.arange(len(wvl)), wvl, 'linear')
    wvl_left = f(left_ips[0])
    
    f = interpolate.interp1d(np.arange(len(wvl)), wvl, 'linear')
    wvl_right = f(right_ips[0])
    
    fwhm = wvl_right - wvl_left
    
    # Done
    return wvl[cwl_idx], fwhm

def hyper2xyz(imhyper, wavelengths):
    '''
        Function to convert a hyperspectral image to XYZ image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imxyz: XYZ image.
    '''
    root = this_path.__file__.replace('this_path.py', '/')
    cmf_data = genfromtxt('%slin2012xyz2e_1_7sf.csv'%root, delimiter=',')

    # Interpolate the wavelengths and x, y, z values
    cmf_data_new = np.zeros((len(wavelengths), 3))
    for idx in range(3):
        interp_func = interpolate.interp1d(cmf_data[:, 0],
                                           cmf_data[:, idx+1],
                                           kind='linear',
                                           fill_value='extrapolate')
        cmf_data_new[:, idx] = interp_func(wavelengths)

    # Find valid indices for converting to RGB image.
    #valid_idx = np.where((wavelengths > min(l)) & (wavelengths < max(l)))

    [H, W, T] = imhyper.shape
    hypermat = imhyper.reshape(H*W, T)

    # Compute XYZ image
    imxyz = np.dot(hypermat, cmf_data_new);
    imxyz = imxyz.reshape(H, W, 3)
    
    return imxyz
    
def hyper2rgb(imhyper, wavelengths, gamma=1, normalize=True):
    '''
        Function to convert a hyperspectral image to RGB image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imrgb: RGB image.
    '''
    imxyz = hyper2xyz(imhyper, wavelengths)

    # Before you convert to rgb, normalize
    if normalize:
        imxyz /= imxyz.max()

    # Compute RGB image from xyz
    imrgb = pow(color.xyz2rgb(imxyz), 1.0/gamma)

    return imrgb

def mat2hs(mat, imdim):
    '''
        Convert a matrix to hyperspectral cube.

        Inputs:
            mat: Matrix input with each column being a spectral signature.
            imdim: Dimensions of each spectral image.

        Outputs:
            cube: 3D hyperspectral cube.
    '''
    [h, w] = imdim
    [_, t] = mat.shape

    return mat.reshape(h, w, t)

def hs2mat(cube):
    '''
        Convert a hyperspectral cube to matrix

        Inputs:
            cube: Hyperspectral cube

        Outputs:
            mat: Matrix with each column being a spectral signature.
    '''
    [h, w, t] = cube.shape

    return cube.reshape(h*w, t)

def hsmedfilt2d(cube, bsize=[5, 5]):
    '''
        Median filter a hyperspectral cube by operating per band.

        Inputs:
            cube: Input cube
            bsize: Box size for median filtering

        Outputs:
            cube_filtered: Filtered cube.
    '''
    for idx in range(cube.shape[2]):
        cube[:, :, idx] = signal.medfilt2d(cube[:, :, idx], bsize)

    return cube

def hsvideowrite(cube, filename, fps=30):
    '''
        Create video out of a hyperspectral cube.

        Inputs:
            cube: Hyperspectal cube.
            filename: Name of output video
            fps: Frame rate of output video

        Outputs:
            None
    '''
    H, W, T = cube.shape

    cubemax = cube.max()

    # Create a new video object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(filename, fourcc, fps, (H, W), False)

    for idx in range(T):
        video.write((255*cube[:, :, idx]/cubemax).astype(np.uint8))

    # Close the video
    video.release()

def load_hsim(foldername):
    '''
        Load hyperspectral images from a set of saved image files.

        Inputs:
            foldername: Name of the folder where images are saved

        Outputs:
            cube: 3D hyperspectral cube
            wvl: Wavelengths.
    '''
    # First load wvl file
    wvl = utils.loadp('%s/wavelengths.pkl'%foldername)
    nbands = len(wvl)

    # Load an image to get dimensions
    im = plt.imread('%s/im0.png'%foldername)[:, :, :3].mean(2)
    H, W = im.shape

    cube = np.zeros((H, W, nbands))

    for idx in range(nbands):
        im = plt.imread('%s/im%d.png'%(foldername, idx))[:, :, :3].mean(2)
        cube[:, :, idx] = im

    return cube, wvl

def save_hsim(cube, foldername, wavelengths):
    '''
        Save hyperspectral bands as individual images in uint16 format .

        Inputs:
            cube: Hyperspectral cube
            foldername: Name of saving folder
            wavelengths: Numpy array of wavelengths.

        Outputs:
            None
    '''

    # First save wavelengths
    utils.savep(wavelengths, '%s/wavelengths.pkl'%foldername)

    # Now save images
    cmax = cube.max()
    vmax = pow(2, 16)-1
    for idx in range(len(wavelengths)):
        im = (vmax*cube[:, :, idx]/cmax).astype(np.uint16)
        plt.imsave('%s/im%d.png'%(foldername, idx),
                   im, cmap='gray', vim=0, vmax=vmax)
        
