#!/usr/bin/env python

'''
    Implements loss functions for neural network-based adaptive HSI sampling
    
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
from skimage.transform import rotate
import matplotlib.pyplot as plt

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class GuidedLoss(nn.Module):
    '''
        This loss function estimates effectiveness of sparse sampling by
        guided filtering the masked input with the input itself
        
        Inputs:
            winsize: 2-tuple window size
            
        Outputs:
            lossval: GuidedLoss value
    '''
    def __init__(self, winsize=(7, 7)):
        # Initialize parent
        super(GuidedLoss, self).__init__()
        
        self.winsize = winsize
        self.eps = 1e-3
        
        self.win_hori = torch.ones(1, 1, 1, winsize[1])
        self.win_vert = torch.ones(1, 1, winsize[0], 1)
        
    def boxfilter(self, X):
        # Convolve
        nchan = X.shape[1]
        
        if X.is_cuda:
            win_hori = self.win_hori.cuda(X.get_device())
            win_vert = self.win_vert.cuda(X.get_device())
        else:
            win_hori = self.win_hori
            win_vert = self.win_vert
            
        win_hori = Variable(win_hori, requires_grad=False)
        win_vert = Variable(win_vert, requires_grad=False)
            
        X_hori = F.conv2d(X, win_hori, padding=(0, self.winsize[1]//2),
                          groups=nchan)
        X_vert = F.conv2d(X_hori, win_vert, padding=(self.winsize[0]//2, 0),
                          groups=nchan)
        
        return X_vert        
        
    def guided_filter(self, guide, img, mask):
        '''
            Guided filtering for first layer using affine approach
        '''
        # Compute individual terms
        im11 = self.boxfilter(mask)

        imxy = self.boxfilter(img*guide*mask)/im11
        imxx = self.boxfilter(guide*guide*mask)/im11

        imx1 = self.boxfilter(guide*mask)/im11
        imy1 = self.boxfilter(img*mask)/im11

        im_alpha = (imxy - imx1*imy1)/(imxx - imx1*imx1 + self.eps)
        im_beta = imy1 - im_alpha*imx1
        
        imrep = torch.repeat_interleave(guide[:, [0], :, :], 1, img.shape[1])

        X = im_alpha*imrep + im_beta

        return X
    
    def forward(self, X, mask):
        '''
            First N-1 channels are image channels, and last channel is mask
        '''
        #X = Variable(X, requires_grad=True).to(X.get_device())

        if X.is_cuda:
            loss = torch.zeros(1, device=X.get_device())
        else:
            loss = torch.zeros(1, device='cpu')
            
        loss = Variable(loss, requires_grad=True)
            
        for chan_idx in range(X.shape[1]):
            imfiltered = self.guided_filter(X[:, [chan_idx], :, :],
                                            X[:, [chan_idx], :, :],
                                            mask)

            loss = loss + torch.mean(torch.square(imfiltered - X[:, [chan_idx], :, :]))
            
        return torch.sqrt(loss)        
        
class SeparationLoss(nn.Module):
    '''
        Loss function for enforcing minimum separation along horizontal
        direction.
        
        Inputs:
            minsep: Minimum separation between two openings
            
        Outputs:
            loss_val: Separation loss
    '''
    def __init__(self, minsep=31):
        super(SeparationLoss, self).__init__()
        
        self.minsep = minsep

        self.window = torch.ones(1, 1, 1, minsep)
        
    def forward(self, X):
        
        if X.is_cuda:
            window = self.window.cuda(device=X.get_device())
        else:
            window = self.window
            
        window = Variable(window, requires_grad=False)
            
        nchan = X.shape[1]
        X_conv = F.conv2d(X, window, padding=(0, self.minsep//2),
                          groups=nchan)
        
        X_conv[X_conv < 1] = 1
        
        loss = torch.mean(torch.abs(X_conv - 1))
        
        return loss