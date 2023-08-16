#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# System imports
import os
import sys
import time
import pickle
import pdb
import glob

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal

# Plotting
import matplotlib.pyplot as plt

def stack2mosaic(imstack):
    '''
        Convert a 3D stack of images to a 2D mosaic

        Inputs:
            imstack: (H, W, nimg) stack of images

        Outputs:
            immosaic: A 2D mosaic of images
    '''
    H, W, nimg = imstack.shape

    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))

    immosaic = np.zeros((H*nrows, W*ncols), dtype=imstack.dtype)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img_idx = row_idx*ncols + col_idx
            if img_idx >= nimg:
                return immosaic

            immosaic[row_idx*H:(row_idx+1)*H, col_idx*W:(col_idx+1)*W] = \
                                              imstack[:, :, img_idx]

    return immosaic

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1))
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def asnr(x, xhat, compute_psnr=False):
    '''
        Compute affine SNR, which accounts for any scaling and shift between two
        signals

        Inputs:
            x: Ground truth signal(ndarray)
            xhat: Approximation of x

        Outputs:
            asnr_val: 20log10(||x||/||x - (a.xhat + b)||)
                where a, b are scalars that miminize MSE between x and xhat
    '''
    mxy = (x*xhat).mean()
    mxx = (xhat*xhat).mean()
    mx = xhat.mean()
    my = x.mean()
    

    a = (mxy - mx*my)/(mxx - mx*mx)
    b = my - a*mx

    if compute_psnr:
        return psnr(x, a*xhat + b)
    else:
        return rsnr(x, a*xhat + b)

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2))

    snrval = 10*np.log10((np.max(x)**2)/denom)

    return snrval

def SAM_3d(x, xhat, avg=False):
    '''
        Compute SAM for a 3D hyperspectral cube

        Inputs:
            x: Ground truth HSI
            xhat: Reconstructed HSI
            avg: If True, average spatially

        Outputs:
            SAM: SAM map (or average value)
    '''

    x_norm = (x*x).sum(2)
    xhat_norm = (xhat*xhat).sum(2)

    xxhat = abs(x*xhat).sum(2)

    SAM = np.arccos(xxhat/np.sqrt(x_norm*xhat_norm + 1e-12))*180/np.pi

    if avg:
        SAM = np.mean(SAM)

    return SAM

def sam(x, xhat):
    xn = (x*x).sum()
    xhatn = (xhat*xhat).sum()
    
    xxhatn = abs(x*xhat).sum()
    
    samval = np.arccos(xxhatn/np.sqrt(xn*xhatn + 1e-12))*180/np.pi
    return samval

def savep(data, filename):
    '''
        Tiny wrapper to store data as a python pickle.

        Inputs:
            data: List of data
            filename: Name of file to save
    '''
    f = open(filename, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def loadp(filename):
    '''
        Tiny wrapper to load data from python pickle.

        Inputs:
            filename: Name of file to load from

        Outputs:
            data: Output data from pickle file
    '''
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def display_time(total_time):
    '''
        Tiny wrapper to print time in an appropriate way.

        Inputs:
            total_time: Raw time in seconds

        Outputs:
            None
    '''
    if total_time < 60:
        print('Total scanning time: %.2f seconds'%total_time)
    elif total_time < 3600:
        print('Total scanning time: %.2f minutes'%(total_time/60))
    elif total_time < 86400:
        print('Total scanning time: %.2f hours'%(total_time/3600))
    else:
        print('Total scanning time: %.2f days'%(total_time/86400))
        print('... what are you really doing?')

def dither(im):
    '''
        Implements Floyd-Steinberg spatial dithering algorithm

        Inputs:
            im: Grayscale image normalized between 0, 1

        Outputs:
            imdither: Dithered image
    '''
    H, W = im.shape
    imdither = np.zeros((H+1, W+1))

    # Pad the last row/column to propagate error
    imdither[:H, :W] = im
    imdither[H, :W] = im[H-1, :W]
    imdither[:H, W] = im[:H, W-1]
    imdither[H, W] = im[H-1, W-1]

    for h in range(0, H):
        for w in range(1, W):
            oldpixel = imdither[h, w]
            newpixel = (oldpixel > 0.5)
            imdither[h, w] = newpixel

            err = oldpixel - newpixel
            imdither[h, w+1] += (err * 7.0/16)
            imdither[h+1, w-1] += (err * 3.0/16)
            imdither[h+1, w] += (err * 5.0/16)
            imdither[h+1, w+1] += (err * 1.0/16)

    return imdither[:H, :W]

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = noise_snr*np.random.randn(x_meas.size).reshape(x_meas.shape)

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x >= 0] = np.random.poisson(x_meas[x >= 0])
        x_meas[x < 0] = -np.random.poisson(-x_meas[x < 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def deconvwnr1(sig, kernel, wconst=1e-2):
    '''
        Deconvolve a 1D signal using Wiener deconvolution

        Inputs:
            sig: Input signal
            kernel: Impulse response
            wconst: Wiener deconvolution constant

        Outputs:
            sig_deconv: Deconvolved signal
    '''

    sigshape = sig.shape
    sig = sig.ravel()
    kernel = kernel.ravel()

    N = sig.size
    M = kernel.size

    # Padd signal to regularize 
    sig_padded = np.zeros(N+2*M)
    sig_padded[M:-M] = sig

    # Compute Fourier transform
    sig_fft = np.fft.fft(sig_padded)
    kernel_fft = np.fft.fft(kernel, n=(N+2*M))

    # Compute inverse kernel
    kernel_inv_fft = np.conj(kernel_fft)/(np.abs(kernel_fft)**2 + wconst)

    # Now compute deconvolution
    sig_deconv_fft = sig_fft*kernel_inv_fft

    # Compute inverse fourier transform
    sig_deconv_padded = np.fft.ifft(sig_deconv_fft)

    # Clip
    sig_deconv = np.real(sig_deconv_padded[M//2:M//2+N])

    return sig_deconv.reshape(sigshape)

def lowpassfilter(data, order=5, freq=0.5):
    '''
        Low pass filter the input data with butterworth filter.
        This is based on Zackory's github repo: 
            https://github.com/Healthcare-Robotics/smm50

        Inputs:
            data: Data to be filtered with each row being a spectral profile
            order: Order of butterworth filter
            freq: Cutoff frequency

        Outputs:
            data_smooth: Smoothed spectral profiles
    '''
    # Get butterworth coefficients
    b, a = signal.butter(order, freq, analog=False)

    # Then just apply the filter
    data_smooth = signal.filtfilt(b, a, data)

    return data_smooth

def smoothen_spectra(data, bsize=10, method='gauss'):
    '''
        Smoothen rows of spectra with some kernel

        Inputs:
            data: nsamples x nwavelengths spectral matrix
            bsize: Size of blur kernel. For gaussian blur, it is FWHM
            method: 'box', 'poly', or 'gauss'. If ply, bsize is the order of
                the polynomial

        Outputs:
            data_smooth: Smoothened data
    '''
    data_smooth = np.zeros_like(data)

    if method == 'box':
        kernel = np.ones(bsize)/bsize
    elif method == 'gauss':
        sigma = bsize/(2*np.sqrt(2*np.log(2)))
        kernlen = int(sigma*12)
        x = np.arange(-kernlen//2, kernlen//2)
        kernel = np.exp(-(x*x)/(2*sigma*sigma))
    else:
        kernel = None

    for idx in range(data.shape[0]):
        data_smooth[idx, :] = np.convolve(data[idx, :], kernel, 'same')

    return data_smooth

def polyfilt(data, polyord=10):
    '''
        Polynomial filter a 1D vector.

        Inputs:
            data: 1D vector which requires smoothening
            polyord: Order of the polynomial to use for fitting

        Outputs:
            data_filt: poly fitted data
    '''
    datashape = data.shape

    data_vec = data.ravel()
    N = data_vec.size

    x = np.arange(N)

    coefs = np.polyfit(x, data_vec, polyord)
    func = np.poly1d(coefs)
    data_fit = func(x)

    return data_fit.reshape(datashape)

def names2labels(label_names, label_dict):
    '''
        Convert a list of label names to an array of labels

        Inputs:
            label_names: List of label names
            label_dict: A dictionary of the form, label_name:label_idx.

        Outputs:
            labels: Array of label indices. The label is -1 if key was not found
                in the dictionary
    '''
    labels = []

    for label_name in label_names:
        if label_name in label_dict.keys():
            labels.append(label_dict[label_name])
        else:
            labels.append(-1)

    return np.array(labels)

class UnitScaler(object):
    '''
        This is a place holder for StandardScaler when scaling is not utilized
    '''
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x

def grid_plot(imdata, clip=False, saveimg=False):
    '''
        Plot 3D set of images into a 2D grid using subplots.

        Inputs:
            imdata: N x H x W image stack

        Outputs:
            None
    '''
    N, H, W = imdata.shape

    nrows = int(np.sqrt(N))
    ncols = int(np.ceil(N/nrows))
    
    if clip:
        imdata = np.clip(imdata, 0, 1)

    for idx in range(N):
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(imdata[idx, :, :], cmap='gray')
        plt.xticks([], [])
        plt.yticks([], [])
        
        if saveimg:
            plt.imsave('img%d.png'%idx, normalize(imdata[idx, ...], True))

def boxify(imrgb, limits, points, color=np.array([1, 1, 1]), color2=None):
    imrgb = np.copy(imrgb)
    
    if color2 is None:
        color2 = color
    
    w1, h1, dw, dh = limits
    w2 = w1 + dw
    h2 = h1 + dh
    
    imrgb[h1:h1+3, w1:w2, :] = color
    imrgb[h2:h2+3, w1:w2, :] = color
    imrgb[h1:h2, w1:w1+3, :] = color
    imrgb[h1:h2, w2:w2+3, :] = color
    
    inset = imrgb[h1:h2, w1:w2, :]
    
    if points is not None:
        x, y = points[0, :]
        imrgb[x-5:x+5, y-5, :] = color2
        imrgb[x-5:x+5, y+5, :] = color2
        imrgb[x-5, y-5:y+5, :] = color2
        imrgb[x+5, y-5:y+5, :] = color2
    
    return imrgb, inset

def get_err_vs_dist(err, mask, nbins=20):
    '''
        Compute error as a function of distance by binning
        
        Inputs:
            err: H x W error image
            mask: Mask with 1 for sampled locations
            nbins: Number of bins for binning distances
            
        Outputs:
            dist_array: Distance from nearest sampling location
            err_array: Error as a function of distance_array
            err_min: Min value for each bin
            err_max: max value for each bin
            err_std: Standard deviation for each bin
    '''
    H, W = mask.shape
    Y, X = np.mgrid[:H, :W]
    xs = X[mask == 1]
    ys = Y[mask == 1]
    
    dists = np.hypot(X.reshape(-1, 1) - xs.reshape(1, -1),
                     Y.reshape(-1, 1) - ys.reshape(1, -1))
    dist_min = dists.min(1).reshape(H, W)
    
    