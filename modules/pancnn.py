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

# CASSI for creating sampling mask
from modules import cassi
from modules import utils
from modules.filtering import guided_filter, gausswin2d

def loadmodel(modelname, modeltype, maskmode=False):
    '''
        Load NN model along with pretrained params
    '''
    state_dict = torch.load('models/%s_model.pth'%modelname)
    
    if modeltype == 'adaptive':
        # Obtain guided filter sizes
        gnames = [key for key in state_dict if 'guide_filters' in key]
        gnames = [gname for gname in gnames if 'weight' in gname]
        
        guide_ksizes = [(state_dict[gname].shape[-2], state_dict[gname].shape[-1]) for gname in gnames]
        
        # Now create network
        net = PanCNNAdaptive(input_dim=1, maskmode=maskmode,
                                    guide_ksizes=guide_ksizes)
    elif modeltype == 'guidedmulti':
        nfilters = [state_dict['pancnnguided.conv1.weight'].shape[0],
                    state_dict['pancnnguided.conv2.weight'].shape[0]]
        guide_ksize = state_dict['pancnnguided.conv1.weight'].shape[-2]
        net = PanCNNMultiGuided(input_dim=guide_ksize, nfilters=nfilters,
                                  maskmode=maskmode)
    elif modeltype=='guided3d':
        guide_ksize = state_dict['boxfilter.weight'].shape[-1]
        nfilters = state_dict['boxfilter.weight'].shape[0]
        stage2_nlayers = len([key for key in state_dict if 'stage2' in key]) // 2
        stage2_nconv = state_dict['stage2.0.weight'].shape[0]
        input_dim = state_dict['stage2.0.weight'].shape[1] // nfilters
        net = PanCNN3DGuided(input_dim=input_dim, nfilters=nfilters,
                                    guided_ksize=(guide_ksize, guide_ksize),
                                    stage2_nlayers=stage2_nlayers,
                                    stage2_nconv=stage2_nconv)
    elif modeltype == 'linear':
        nfilters = [state_dict['conv1.weight'].shape[0], 32]
        guide_ksize = state_dict['conv1.weight'].shape[-2]
        net = PanCNNLinear(input_dim=guide_ksize, nfilters=nfilters,
                                  maskmode=maskmode)
    elif modeltype == 'refine':
        nfilters = [state_dict['conv1.weight'].shape[0],
                    state_dict['conv2.weight'].shape[0]]
        guide_ksize = state_dict['conv1.weight'].shape[-2]
        net = PanCNNRefine(input_dim=guide_ksize, nfilters=nfilters,
                                  maskmode=maskmode)
    else:
        nfilters = [state_dict['conv1.weight'].shape[0],
                    state_dict['conv2.weight'].shape[0]]
        guide_ksize = state_dict['conv1.weight'].shape[-2]
        net = PanCNNGuided(input_dim=guide_ksize, nfilters=nfilters,
                                  maskmode=maskmode)
        
        
    # Load parameters
    net.load_state_dict(state_dict)
    net.eval()
    return net

class PanCNNMultiGuided(nn.Module):
    '''
        Neural network module for 3D hyperspectral reconstruction in a per-slice
        manner.
        
        The module is basically a wrapper around PanCNNGuided
    '''
    def __init__(self, input_dim=31, dropout=0.1, padding=True,
                 nfilters=[8, 32], maskmode=False):
        self.input_dim = input_dim

        # Constant for stable division
        self.eps = 1e-3
        self.maskmode = maskmode

        # Initialize nn.Module
        super(PanCNNMultiGuided, self).__init__()

        self.pancnnguided = PanCNNGuided(input_dim, dropout,
                                         padding=padding, nfilters=nfilters,
                                         maskmode=maskmode)
        
    def forward(self, X):
        '''
            Forward operator
            X size: batchsize x (nlambda + 1) x H x W, where the last channel
                is the guide image
        '''
        _, nchan, H, W = X.shape
        
        # Stage 1 -- guided filtering
        guided_list = []
       
        if self.maskmode:
            nwvl = nchan-2
        else:
            nwvl = nchan-1

        for idx in range(nwvl):
            if self.maskmode:
                inputs = X[:, [idx, -2, -1], :, :]
            else:
                inputs = X[:, [idx, -1], :, :]
            guided_list.append(self.pancnnguided(inputs))
        
        return torch.cat(guided_list, 1)
        
class PanCNN3DGuided(nn.Module):
    '''
        Neural network module for 3D hyperspectral reconstruction
        
        Inputs:
            input_dim: Number of wavelength bands
            dropout: Dropout fraction
            guide_ksize: Filter size for guided filtering -- stage1 -- space only
            
            stage2_nlayers: Number of stage2 layers for 3D convolution
            stage2_nconv: Number of filters per layer
            
            stage3_nlayers: Number of stage3 layers for spectrum-only
            stage3_nconv: number of filters per layer
            
        Outputs:
            See nn.Module
    '''
    def __init__(self, input_dim, dropout=0.0, guided_ksize=(11, 11),
                 nfilters=1, stage2_nlayers=4, stage2_nconv=4,
                 maskmode=False):
        
        # Set all module parameters
        self.input_dim = input_dim
        self.guide_ksize = guided_ksize
        
        # Constant for stable division
        self.eps = 1e-3
        
        # Padding mode for convolution
        padding_mode = 'reflect'
        
        # For now, disable mask mode
        self.maskmode = maskmode

        # Initialize nn.Module
        super(PanCNN3DGuided, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Create box filter for guided convolution
        padding = (guided_ksize[0] // 2, guided_ksize[1] // 2)
        self.boxfilter = nn.Conv2d(1, nfilters, guided_ksize, padding=padding,
                                   padding_mode=padding_mode)
        
        # Initialize boxfilter with all ones -- helps converge faster
        with torch.no_grad():
            self.boxfilter.weight.fill_(1.0)
        
        # Generate stage 1
        stage2_list = [nn.Conv2d(input_dim*nfilters, stage2_nconv, (3, 3),
                                 padding=(1, 1), padding_mode=padding_mode),
                       self.relu]
        
        for _ in range(stage2_nlayers - 2):
            stage2_list.append(nn.Conv2d(stage2_nconv, stage2_nconv, (3, 3),
                                         padding=(1, 1),
                                         padding_mode=padding_mode))
            stage2_list.append(self.relu)
            
        stage2_list.append(nn.Conv2d(stage2_nconv, input_dim, (3, 3),
                                     padding=(1, 1),
                                 padding_mode=padding_mode))
        
        self.stage2 = nn.Sequential(*stage2_list)
        
    def forward(self, X, **kwargs):
        '''
            Forward operator
            X size: batchsize x (nlambda + 1) x H x W, where the last channel
                is the guide image
        '''
        _, nchan, H, W = X.shape
        
        # Stage 1 -- guided filtering
        guided_list = []
       
        if self.maskmode:
            nwvl = nchan - 2
        else:
            nwvl = nchan - 1

        for idx in range(nwvl):            
            if self.maskmode:
                inputs = X[:, [idx, -2, -1], :, :]
            else:
                inputs = X[:, [idx, -1], :, :]
                
            minval = inputs.min()
            maxval = inputs.max()
            
            inputs = (inputs - minval)/(maxval - minval)
            output = guided_filter(inputs,
                                   self.boxfilter,
                                   maskmode=self.maskmode,
                                   eps=self.eps)
            # x2 = (x1 - minval)/(maxval - minval)
            # x1 = (maxval - minval)*x1 + minval
            output = (maxval - minval)*output + minval
            guided_list.append(output)
        self.Xguided = torch.cat(guided_list, 1)
        
        # Stage 2 -- 2D convolution
        self.X2 = self.stage2(self.Xguided)
        
        return self.X2

class FlowNet(nn.Module):
    '''
        Network to estimate flow field for adaptive sampling mask
        
        Inputs:
            guide_ksize: 2-tuple guide filter size
            nlayers: Number of convolutional layers
            nconv: Number of convolutions per layer
            flowrange: 2-tuple of min and max flow
            
        Outputs:
            see nn.Module
    '''
    def __init__(self, guide_ksize=(11, 11), nlayers=4, nconv=16,
                 flowrange=[-5, 5]):
        self.guide_ksize = guide_ksize
        self.nlayers = nlayers
        self.nconv = nconv
        self.flowrange = flowrange
        self.eps = 1e-3
        
        # Initialze super class
        super(FlowNet, self).__init__()
        
        # Create functional units
        self.relu = nn.ReLU(inplace=True)
        
        # Create learnable box filter
        padding = (guide_ksize[0]//2, guide_ksize[1]//2)
        self.boxfilter = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=guide_ksize,
                                   padding=padding)
        
        modules_list = [nn.Conv2d(3, self.nconv, (3, 3),
                                  padding=(1, 1)), self.relu]
        
        # Now create convolutional layers
        for idx in range(nlayers):
            modules_list.append(nn.Conv2d(self.nconv, self.nconv,
                                         (3, 3), padding=(1, 1)))
            modules_list.append(self.relu)
            
        # Tail is an output of two channels -- x flow and y flow
        modules_list.append(nn.Conv2d(self.nconv, 2, (3, 3), padding=(1, 1)))
        
        self.stage2 = nn.Sequential(*modules_list)
        
        # Initialize box filter
        torch.nn.init.ones_(self.boxfilter.weight)
        
    def guided_filter(self, guide, img, mask):
        '''
            Guided filtering for first layer using affine approach
        '''
        # Compute individual terms
        _, _, H, W = img.size()
        im11 = self.boxfilter(mask)

        imxy = self.boxfilter(img*guide*mask)/im11
        imxx = self.boxfilter(guide*guide*mask)/im11

        imx1 = self.boxfilter(guide*mask)/im11
        imy1 = self.boxfilter(img*mask)/im11

        im_alpha = (imxy - imx1*imy1)/(imxx - imx1*imx1 + self.eps)
        im_beta = imy1 - im_alpha*imx1

        imrep = torch.repeat_interleave(guide[:, [0], :, :], 1, 1)

        X = im_alpha*imrep + im_beta

        return X
    
    def forward(self, X):
        '''
            Forward operator
            
            X structure:
            X[:, 0, ...] -- red channel
            X[:, 1, ...] -- green channel
            X[:, 2, ...] -- blue channel
            X[:, 3, ...] -- mask
        '''
        _, _, H, W = X.size()
        
        # Generate panchromatic image for guidance
        guide = X[:, [0, 1, 2], :, :].mean(1, keepdim=True)
        mask = X[:, [3], :, :]
        
        # First guide filter the masked images
        imr = self.guided_filter(guide, X[:, [0], :, :], X[:, [3], ...])
        img = self.guided_filter(guide, X[:, [1], :, :], X[:, [3], ...])
        imb = self.guided_filter(guide, X[:, [2], :, :], X[:, [3], ...])
        
        # Now concatenate the outputs
        imrec = torch.cat((imr, img, imb), dim=1)
        self.imerr = torch.abs(imrec - X[:, [0, 1, 2]])
        
        # Stage 2
        X = self.stage2(self.imerr)
        
        # Round the results and normalize to [-1, 1] coordinate system
        X[:, 0, :, :] = torch.round(X[:, 0, :, :]*W/2)*(2/W)
        X[:, 1, :, :] = torch.round(X[:, 1, :, :]*H/2)*(2/H)
        
        flow = torch.clamp(X, self.flowrange[0], self.flowrange[1])
                    
        # Generate meshgrid
        if X.is_cuda:
            device = X.get_device()
        else:
            device = torch.device('cpu')
            
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        
        Ygrid, Xgrid = torch.meshgrid(y, x)
        grid = torch.cat((Xgrid[None, :, :, None], Ygrid[None, :, :, None]), 3)
        grid = Variable(grid, requires_grad=False)
        
        # Generate shifted grid
        shifted_grid = grid + flow.permute(0, 2, 3, 1)
        
        # Now interpolate
        shifted_mask = F.grid_sample(mask, shifted_grid,
                                    mode='nearest', align_corners=True)
        
        return shifted_mask

class PanCNNAdaptive(nn.Module):
    '''
        Spatially adaptive CNN for reconstructing HSI bands from
        sparsely sampled HSI image and a grayscale guide image
        
        Inputs:
            input_dim: Number of wavelength bands (defunct)
            dropout: Dropout fraction
            padding: If True, pad to ensure same sized output (defunct)
            stage1_nlayers: Number of stage1 layers for attention
            stage1_nconv: Number of filters per layer
            guide_ksizes: Filter sizes for guided filtering
            stage3_nlayers: Number of stage3 layers for combination
            stage3_nconv: Number of filters per layer
            
        Output:
            See torch.nn.Module
    '''
    def __init__(self, input_dim=1, dropout=0.1, padding=True, maskmode=False,
                 stage1_nlayers=4, stage1_nconv=16,
                 guide_ksizes=[5, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                 stage3_nlayers=4, stage3_nconv=16):
        
        # Number of input dimensions is 1
        self.input_dim = input_dim
        self.output_dim = 1
        self.isrgb = False
        self.maskmode = maskmode
        self.stage1_nconv = stage1_nconv
        self.stage1_nlayers = stage1_nlayers
        self.guide_ksizes = guide_ksizes
        self.stage3_nconv = stage3_nconv
        self.stage3_nlayers = stage3_nlayers
        
        # Constant for stable division
        self.eps = 1e-7

        # Initialize nn.Module
        super(PanCNNAdaptive, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 -- bunch of convolutions
        # NOTE: The tail should not have a relu as the weights can be positive
        # or negative
        stage1_list = [nn.Conv2d(self.input_dim + 1, self.stage1_nconv, (3, 3),
                                padding=(1, 1), padding_mode='reflect'),
                       self.relu]
        tail = [nn.Conv2d(self.stage1_nconv, len(guide_ksizes), (3, 3),
                          padding=(1, 1), padding_mode='reflect')]

        for idx in range(stage1_nlayers):
            stage1_list.append(nn.Conv2d(self.stage1_nconv, self.stage1_nconv,
                                         (3, 3), padding=(1, 1),
                                         padding_mode='reflect'))
            stage1_list.append(self.relu)

        stage1_list += tail

        self.stage1 = nn.Sequential(*stage1_list)

        # Stage 2 -- filters for guided filtering
        stage2_list = [nn.Conv2d(self.input_dim, self.input_dim,
                                 ksize,
                                 padding=(ksize[0] // 2, ksize[1] // 2),
                                 padding_mode='reflect')\
                       for ksize in guide_ksizes]

        self.guide_filters = nn.ModuleList(stage2_list)

        # Stage 3 -- Final bunch of convolutions
        stage3_list = [nn.Conv2d(len(guide_ksizes), self.stage3_nconv, (3, 3),
                                 padding=(1, 1), padding_mode='reflect'),
                       self.relu]
        tail = [nn.Conv2d(self.stage3_nconv, self.output_dim, (3, 3),
                          padding=(1, 1), padding_mode='reflect')]

        for idx in range(stage3_nlayers):
            stage3_list.append(nn.Conv2d(self.stage3_nconv, self.stage3_nconv,
                                         (3, 3), padding=(1, 1),
                                         padding_mode='reflect'))
            stage3_list.append(self.relu)

        stage3_list += tail

        self.stage3 = nn.Sequential(*stage3_list)
    
    def initialize(self, method='ones'):
        '''
            Initialize weights
        '''
        if method == 'ones':
            for mod in self.guide_filters:
                if type(mod) == nn.Conv2d:
                    torch.nn.init.ones_(mod.weight)

        elif method == 'gaussian':
            # Initialize with rotated gaussian kernels
            for mod in self.guide_filters:
                if type(mod) == nn.Conv2d:
                    _, _, H, W = mod.weight.shape
                    window = gausswin2d((H, W)).astype(np.float32)
                    mod.weight[...] = torch.tensor(window)[None, None, ...]

        elif method == 'gabor':
            # Generate uniformly sampled Gabor filters
            sigma = self.ksize[0][0]/6.0
            nlambd = int(np.sqrt(self.nfilters[0]))
            ntheta = nlambd

            assert nlambd**2 == self.nfilters[0], "First set of filters must be a square"

            lambdas = np.linspace(self.ksize[0][0]/3, self.ksize[0][0], nlambd)
            thetas = np.linspace(0, np.pi/2, ntheta)

            conv1_init = np.zeros((self.nfilters[0], 1, self.ksize[0][0],
                                   self.ksize[0][1]), dtype=np.float32)

            cnt = 0
            for lambd in lambdas:
                for theta in thetas:
                    conv1_init[cnt, 0, :, :] = cv2.getGaborKernel(self.ksize[0],
                                                                  sigma,
                                                                  theta,
                                                                  lambd,
                                                                  1, 0).real

            conv1_weight = torch.tensor(utils.normalize(conv1_init, True))

            self.conv1.weight = torch.nn.Parameter(conv1_weight)

    def forward(self, X, **kwargs):
        '''
            Forward operator
        '''
        # Stage 1 -- attention
        if self.maskmode:
            X1 = self.stage1(X[:, [0, 1], :, :])
        else:
            X1 = self.stage1(X)

        # Stage 2 -- guided filtering for all filter sizes
        X2 = []

        for mod in self.guide_filters:
            X2.append(guided_filter(X, mod, self.maskmode, self.eps))

        X2 = torch.cat(X2, 1)

        # Stage 3 -- weight
        X3 = X1*X2

        # Step 4 -- combine
        X = self.stage3(X3)

        return X

class PanCNNGuided(nn.Module):
    '''
        Simple 3-layer CNN for reconsstructing HSI bands from sparse sampling
        and a grayscale guide image
    '''
    def __init__(self, input_dim=31, dropout=0.1, padding=True,
                 nfilters=[8, 32], maskmode=False):
        self.input_dim = input_dim

        # Constant for stable division
        self.eps = 1e-3
        self.maskmode = maskmode

        # Initialize nn.Module
        super(PanCNNGuided, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Hard coded constants for now
        self.nfilters = nfilters
        self.ksize = [(input_dim, 2*input_dim+1), (3, 3), (5, 5)]

        # Switch this on only if you are adamant about having same size as
        # input. There is some chance padding causes boundary artifacts which
        # affects convergence
        if padding:
            self.padding = [(input_dim//2, input_dim), (1, 1), (2, 2)]
        else:
            self.padding = [(0, 0), (0, 0), (0, 0)]

        # Now create modules
        self.conv1 = nn.Conv2d(1, self.nfilters[0], self.ksize[0],
                               padding=self.padding[0])
        self.conv2 = nn.Conv2d(self.nfilters[0], self.nfilters[1],
                               self.ksize[1], padding=self.padding[1])
        self.conv3 = nn.Conv2d(self.nfilters[1], 1,
                               self.ksize[2], padding=self.padding[2])

    def initialize(self, method='ones'):
        '''
            Initialize weights
        '''
        if method == 'ones':
            torch.nn.init.ones_(self.conv1.weight)

        elif method == 'gaussian':            
            H = self.input_dim
            W = 2*self.input_dim + 1
            window = gausswin2d((H, W)).astype(np.float32)
            self.conv1.weight[...] = torch.tensor(window)[None, None, ...]

        elif method == 'zeros':
            torch.nn.init.zeros_(self.conv1.weight)
            torch.nn.init.zeros_(self.conv2.weight)
            torch.nn.init.zeros_(self.conv3.weight)

        elif method == 'normal':
            torch.nn.init.xavier_normal_(self.conv1.weight)
            torch.nn.init.xavier_normal_(self.conv2.weight)
            torch.nn.init.xavier_normal_(self.conv3.weight)

        elif method == 'uniform':
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)

        elif method == 'identity':
            # Set first "input_dim" channels to identity
            self.conv1.weight[:self.input_dim, :, :, :] = 0
            self.conv2.weight[:, :self.input_dim, :, :] = 0

            for idx in range(self.input_dim):
                self.conv1.weight[idx, idx, 4, 4] = 1
                self.conv2.weight[idx, idx, 2, 2] = 1

    def forward(self, X, **kwargs):
        '''
            Forward operator
        '''
        # Step 1 -- perform optimal operation for guided filtering
        # X[:, [0], :, :] -> spectral band
        # X[:, [1], :, :] -> panchromatic image
        # X[:, [2], :, :] -> mask
        
        X = guided_filter(X, self.conv1, self.maskmode, self.eps)

        # Rest of the operations are similar
        X = self.conv2(X)
        X = self.relu(X)
        X = self.dropout(X)

        # Final layer has only linear operation
        X = self.conv3(X)

        return X
    
class PanCNNLinear(nn.Module):
    '''
        Simple 3-layer CNN for reconsstructing HSI bands from sparse sampling
        and a grayscale guide image
    '''
    def __init__(self, input_dim=31, nfilters=[8, 32], maskmode=False):
        self.input_dim = input_dim

        # Constant for stable division
        self.eps = 1e-3
        self.maskmode = maskmode

        # Initialize nn.Module
        super(PanCNNLinear, self).__init__()

        # Hard coded constants for now
        self.nfilters = nfilters
        self.ksize = [(input_dim, 2*input_dim+1), (3, 3), (5, 5)]

        # Switch this on only if you are adamant about having same size as
        # input. There is some chance padding causes boundary artifacts which
        # affects convergence
        self.padding = [(input_dim//2, input_dim), (1, 1), (2, 2)]

        # Now create modules
        self.conv1 = nn.Conv2d(1, self.nfilters[0], self.ksize[0],
                               padding=self.padding[0])

    def initialize(self, method='ones'):
        '''
            Initialize weights
        '''
        if method == 'ones':
            torch.nn.init.ones_(self.conv1.weight)

        elif method == 'gaussian':            
            H = self.input_dim
            W = 2*self.input_dim + 1
            window = gausswin2d((H, W)).astype(np.float32)
            self.conv1.weight[...] = torch.tensor(window)[None, None, ...]
            
    def forward(self, X, **kwargs):
        '''
            Forward operator
        '''
        # Step 1 -- perform optimal operation for guided filtering
        # X[:, [0], :, :] -> spectral band
        # X[:, [1], :, :] -> panchromatic image
        # X[:, [2], :, :] -> mask
        
        X = guided_filter(X, self.conv1, self.maskmode, self.eps)

        return X.mean(1, keepdim=True)

class PanCNNRefine(nn.Module):
    '''
        Simple 3-layer CNN for reconsstructing HSI bands from sparse sampling
        and a grayscale guide image
    '''
    def __init__(self, input_dim=31, dropout=0.1, padding=True,
                 nfilters=[8, 32], maskmode=False):
        self.input_dim = input_dim

        # Constant for stable division
        self.eps = 1e-3
        self.maskmode = maskmode

        # Initialize nn.Module
        super(PanCNNRefine, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Hard coded constants for now
        self.nfilters = nfilters
        self.ksize = [(input_dim, 2*input_dim+1), (3, 3), (5, 5)]

        # Switch this on only if you are adamant about having same size as
        # input. There is some chance padding causes boundary artifacts which
        # affects convergence
        if padding:
            self.padding = [(input_dim//2, input_dim), (1, 1), (2, 2)]
        else:
            self.padding = [(0, 0), (0, 0), (0, 0)]

        # Now create modules
        self.conv1 = nn.Conv2d(1, self.nfilters[0], self.ksize[0],
                               padding=self.padding[0], bias=False)
        self.conv2 = nn.Conv2d(self.nfilters[0], self.nfilters[1],
                               self.ksize[1], padding=self.padding[1])
        self.conv3 = nn.Conv2d(self.nfilters[1], 1,
                               self.ksize[2], padding=self.padding[2])
        
        # Remove conv1 from training and force them to be all gaussian
        self.conv1.weight.requires_grad = False
        with torch.no_grad():
            self.initialize(method='gaussian')            

    def initialize(self, method='ones'):
        '''
            Initialize weights
        '''
        if method == 'ones':
            torch.nn.init.ones_(self.conv1.weight)

        elif method == 'gaussian':            
            H = self.input_dim
            W = 2*self.input_dim + 1
            window = gausswin2d((H, W)).astype(np.float32)
            self.conv1.weight[...] = torch.tensor(window)[None, None, ...]

        elif method == 'zeros':
            torch.nn.init.zeros_(self.conv1.weight)
            torch.nn.init.zeros_(self.conv2.weight)
            torch.nn.init.zeros_(self.conv3.weight)

        elif method == 'normal':
            torch.nn.init.xavier_normal_(self.conv1.weight)
            torch.nn.init.xavier_normal_(self.conv2.weight)
            torch.nn.init.xavier_normal_(self.conv3.weight)

        elif method == 'uniform':
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)

        elif method == 'identity':
            # Set first "input_dim" channels to identity
            self.conv1.weight[:self.input_dim, :, :, :] = 0
            self.conv2.weight[:, :self.input_dim, :, :] = 0

            for idx in range(self.input_dim):
                self.conv1.weight[idx, idx, 4, 4] = 1
                self.conv2.weight[idx, idx, 2, 2] = 1

    def forward(self, X, **kwargs):
        '''
            Forward operator
        '''
        # Step 1 -- perform optimal operation for guided filtering
        # X[:, [0], :, :] -> spectral band
        # X[:, [1], :, :] -> panchromatic image
        # X[:, [2], :, :] -> mask
        
        X = guided_filter(X, self.conv1, self.maskmode, self.eps)

        # Rest of the operations are similar
        X = self.conv2(X)
        X = self.relu(X)
        X = self.dropout(X)

        # Final layer has only linear operation
        X = self.conv3(X)

        return X
class PanCNN(nn.Module):
    '''
        Simple 3-layer CNN for panchromatic sharpening
    '''
    def __init__(self, input_dim=31, dropout=0.5, padding=True,
                 nfilters=[56, 32]):
        self.input_dim = input_dim

        # Initialize nn.Module
        super(PanCNN, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)

        # Hard coded constants for now
        self.nfilters = nfilters
        self.ksize = [(9, 9), (3, 3), (5, 5)]

        # Switch this on only if you are adamant about having same size as
        # input. There is some chance padding causes boundary artifacts which
        # affects convergence
        if padding:
            self.padding = [(4, 4), (1, 1), (2, 2)]
        else:
            self.padding = [(0, 0), (0, 0), (0, 0)]

        # Now create modules
        self.conv1 = nn.Conv2d(self.input_dim, self.nfilters[0], self.ksize[0],
                               padding=self.padding[0])
        self.bn1 = nn.BatchNorm2d(num_features=self.nfilters[0])

        self.conv2 = nn.Conv2d(self.nfilters[0], self.nfilters[1],
                               self.ksize[1], padding=self.padding[1])
        self.bn2 = nn.BatchNorm2d(num_features=self.nfilters[1])

        self.conv3 = nn.Conv2d(self.nfilters[1], self.input_dim-1,
                               self.ksize[2], padding=self.padding[2])

    def initialize(self, method='ones'):
        '''
            Initialize weights
        '''
        if method == 'ones':
            torch.nn.init.ones_(self.conv1.weight)
            torch.nn.init.ones_(self.conv2.weight)
            torch.nn.init.ones_(self.conv3.weight)

        elif method == 'zeros':
            torch.nn.init.zeros_(self.conv1.weight)
            torch.nn.init.zeros_(self.conv2.weight)
            torch.nn.init.zeros_(self.conv3.weight)

        elif method == 'normal':
            torch.nn.init.kaiming_normal_(self.conv1.weight)
            torch.nn.init.kaiming_normal_(self.conv2.weight)
            torch.nn.init.kaiming_normal_(self.conv3.weight)

        elif method == 'uniform':
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)

        elif method == 'identity':
            # Set first "input_dim" channels to identity
            f1_h = signal.gaussian(self.ksize[0][0], 2/3)
            f1_w = signal.gaussian(self.ksize[0][1], 2/3)
            f1 = f1_h.reshape(-1, 1).dot(f1_w.reshape(1, -1))

            f2_h = signal.gaussian(self.ksize[1][0], 2/3)
            f2_w = signal.gaussian(self.ksize[1][1], 2/3)
            f2 = f2_h.reshape(-1, 1).dot(f2_w.reshape(1, -1))

            f3_h = signal.gaussian(self.ksize[2][0], 2/3)
            f3_w = signal.gaussian(self.ksize[2][1], 2/3)
            f3 = f3_h.reshape(-1, 1).dot(f3_w.reshape(1, -1))

            f1 = f1/f1.sum()
            f2 = f2/f2.sum()
            f3 = f3/f3.sum()

            self.conv1.weight[:, :, :, :] = torch.tensor(f1)[None, None, :, :]
            self.conv2.weight[:, :, :, :] = torch.tensor(f2)[None, None, :, :]
            self.conv3.weight[:, :, :, :] = torch.tensor(f3)[None, None, :, :]

    def forward(self, X, **kwargs):
        '''
            Forward operator
        '''
        # Three 2D convolutions and two ReLU operators
        X = self.conv1(X)
        #X = self.bn1(X)
        X = F.relu(X)
        X = self.dropout(X)

        X = self.conv2(X)
        #X = self.bn2(X)
        X = F.relu(X)
        X = self.dropout(X)

        # Final layer has only linear operation
        X = self.conv3(X)

        return X

if __name__ == '__main__':
    model = PanCNN3DGuided(input_dim=31)
    data = io.loadmat('../../power_hs/data/kaist3_clipped/kaist3_clipped.mat')
    hypercube = data['hypercube'].astype(np.float32)[:128, :128, :]
    imrgb = data['imrgb'].astype(np.float32)[:128, :128, :]/255

    hypercube = torch.tensor(hypercube).permute([2, 0, 1])[None, ...]
    imrgb = torch.tensor(imrgb).permute([2, 0, 1])[None, ...]
    
    imgray = imrgb.mean(1, keepdim=True)
    
    inputs = torch.cat((hypercube, imgray), 1)

    filters_output = model(inputs)
