#!/usr/bin/env python

'''
    Implements model for panchromatic sharpening with CNNs. This implementation
    is mostly a reinterpretation of the following:

    "Pansharpening by Convolutional Neural Networks",
    written by  G. Masi, D. Cozzolino, L. Verdoliva and G. Scarpa,
    Remote Sensing, 2016.
'''

# System imports
import os
import sys
import argparse
import configparser
import ast
import pdb

# Scientific computing
import numpy as np
from scipy import linalg as lin
from scipy import io
from scipy import signal
from scipy import ndimage
from skimage.transform import rotate
import matplotlib.pyplot as plt

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# Opencv for imresize
import cv2

def guided_filter(X, boxfilter, maskmode=False, eps=1e-3):
    '''
        Generalized guided filtering
        
        X: batch x nchan x H x W input tensor. Indices 0, 1 are image to be
            filtered and guide image respectively.
            
            If maskmode is True, guided filtering is done with weighted least
            squares. Else it is usual guided filtering.
        boxfilter: nn.Conv2d instantiation
        maskmode: If true, perform mask weighted least squares.
        eps: Stabilization constant
        
        Outputs:
            imfiltered: batch x 1 x H x W output filtered image.
    '''
    img = X[:, [0], :, :]
    guide = X[:, [1], :, :]
    
    if maskmode:    
        mask = X[:, [2], :, :]
    else:
        _, _, H, W = img.size()
        mask = Variable(img.data.new().resize_((1, 1, H, W)).fill_(1.0),
                        requires_grad=False)
    im11 = boxfilter(mask)

    imxy = boxfilter(img*guide*mask)/im11
    imxx = boxfilter(guide*guide*mask)/im11

    imx1 = boxfilter(guide*mask)/im11
    imy1 = boxfilter(img*mask)/im11

    # NOTE: Division, for some reason is unstable, and leads to negative values
    # even if the inputs are all positive. To counter this, we have to do a 
    # relu on the numerator and denominator
    im_alpha = F.relu(imxy - imx1*imy1)/(F.relu(imxx - imx1*imx1) + eps)
    im_beta = imy1 - im_alpha*imx1

    _, output_dim, _, _ = im_beta.size()
    imrep = torch.repeat_interleave(guide, output_dim, 1)

    X = im_alpha*imrep + im_beta

    return X

def gausswin2d(winsize, sigma=None):
    '''
        Create 2D gaussian window
        
        Inputs:
            winsize: 2D size of the window
            sigma: 2D standard deviations. If None, assume it to be winsize/6
            
        Outputs:
            gausswin: 2D Gaussian window
    '''
    [H, W] = winsize

    if sigma is None:
        sh = H/3
        sw = W/3
    else:
        sh, sw = sigma

    hwin = signal.windows.gaussian(H, sh).reshape(H, 1)
    wwin = signal.windows.gaussian(W, sw).reshape(1, W)
    
    gausswin = hwin.dot(wwin)
    
    return gausswin

def diskwin(rad):
    '''
        Create disk window. Largely similar to matlab 'fspecial.m' 'disk' option
        
        Inputs:
            rad: Disk radius
        
        Outputs:
           window: Disk window 
    '''
    crad  = int(np.ceil(rad-0.5))
    [x,y] = np.meshgrid(np.arange(-crad, crad+1),np.arange(-crad, crad+1))
    
    maxxy = np.maximum(abs(x),abs(y));
    minxy = np.minimum(abs(x),abs(y));
    
    m1 = (rad**2 <  (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) + \
        (rad**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2)* \
        np.sqrt(rad**2 - (maxxy + 0.5)**2)
    m2 = (rad**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) + \
        (rad**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)* \
        np.sqrt(rad**2 - (maxxy - 0.5)**2)
    sgrid = ((rad**2)*(0.5*(np.arcsin(m2/rad) - np.arcsin(m1/rad)) + \
        0.25*(np.sin(2*np.arcsin(m2/rad)) - np.sin(2*np.arcsin(m1/rad)))) - \
        (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5)) \
        *((((rad**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & \
        (rad**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | \
        ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy+0.5>=rad))))
        
    sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < rad**2)
    
    sgrid[crad+1,crad+1] = min(np.pi*(rad**2),np.pi/2)
    
    if ((crad>0) and (rad > crad-0.5) and (rad**2 < (crad-0.5)**2+0.25)):
        m1  = np.sqrt(rad**2 - (crad - 0.5)**2)
        m1n = m1/rad
        sg0 = 2*(rad^2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
        sgrid[2*crad,crad] = sg0
        sgrid[crad,2*crad] = sg0
        sgrid[crad,0]        = sg0
        sgrid[0,crad]        = sg0
        sgrid[2*crad-1,crad]   = sgrid[2*crad-1,crad] - sg0
        sgrid[crad,2*crad-1]   = sgrid[crad,2*crad-1] - sg0
        sgrid[crad,1]        = sgrid[crad,1]      - sg0
        sgrid[1,crad]        = sgrid[1,crad]      - sg0
    
    sgrid[crad,crad] = min(sgrid[crad,crad],1);
    
    return sgrid
    
    #return sgrid/sgrid.sum()
    
def gaussdiskwin(rad):
    '''
        Create disk window using Gaussian kernel.
        
        Inputs:
            rad: Disk radius
        
        Outputs:
           window: Disk window 
    '''
    radint = int(np.ceil(rad - 0.5))
    winsize = 2*radint + 1
    
    sigma = winsize/(2*np.sqrt(2*np.log(2)))
        
    return gausswin2d([winsize, winsize], [sigma, sigma])

def convsep(im, winsize=[31, 31], wintype='gaussian'):
    '''
        Implement separable 2D convolution for guided filtering.

        Inputs:
            im: Image to filter
            winsize: Size of window. Default is [31, 31]
            wintype: 'gaussian' or ones
    '''
    if wintype == 'gaussian':
        win_h = signal.gaussian(winsize[0], winsize[0]/3)
        win_w = signal.gaussian(winsize[1], winsize[1]/3)
    else:
        win_h = np.ones(winsize[0])
        win_w = np.ones(winsize[1])

    im_h = signal.convolve2d(im, win_h.reshape(-1, 1), mode='same')
    im_w = signal.convolve2d(im_h, win_w.reshape(1, -1), mode='same')

    return im_w

def guidedFilterMasked(guide, img, mask, winsize=7, eps=1e-6, wintype='gaussian'):
    '''
        Reconstruct an image using masked guided filtering approach
        
        Inputs:
            guide: Guide image
            img: Image to be filtered (possibly masked)
            mask: Binary mask
            winsize: Size of convolution
            eps: Stabilization constant
            wintype: Gaussian or box window
            
        Outputs:
            filtered_img: Filtered image
    '''
    # Compute individual terms
    H, W = guide.shape
    im11 = convsep(mask, (winsize, winsize), wintype)

    imxy = convsep(guide*img*mask, (winsize, winsize), wintype)/im11
    imxx = convsep(guide*guide*mask, (winsize, winsize), wintype)/im11

    imx1 = convsep(guide*mask, (winsize, winsize), wintype)/im11
    imy1 = convsep(img*mask, (winsize, winsize), wintype)/im11

    im_alpha = (imxy - imx1*imy1)/(imxx - imx1*imx1 + eps)
    im_beta = imy1 - im_alpha*imx1
    
    return im_alpha*guide + im_beta

if __name__ == '__main__':
    window = gaussdiskwin(1.7)
    
    plt.imshow(window); plt.show()