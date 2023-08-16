#!/usr/bin/env python3

'''
    Efficient Pytorch implementation of Superpixel Samping Networks (SSN) by
    Varun Jampani and others.

    Paper: https://varunjampani.github.io/papers/jampani18_SSN.pdf
    Original Caffe code: https://github.com/NVlabs/ssn_superpixels

    Note that we are not yet implementing connectivity.
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
from skimage import color
from skimage import segmentation
from skimage.transform import rotate
import matplotlib.pyplot as plt

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# cupy
import cupy
from cupyx.scipy import ndimage

# Opencv for imresize
import cv2

#import cassi_cp

class SSN(nn.Module):
    '''
        Convolutional features and superpixel iterations implementation
        
        Inputs:
            n_superpixels_row: Number of rows of superpixels
            n_superpixels_col: Number of colums of superpixels
            compactness: Weight given to regular shape of superpixels
            stage1_nlayers: Number of layers for stage1
            stage1_nconv: Number of convolutions per layer
            device: torch device (required for sending some variables)
            
        Outputs:
            See nn.Module
    '''
    def __init__(self, niters=10, n_superpixels_row=5, n_superpixels_col=5,
                 compactness=10.0, stage1_nlayers=5, stage1_nconv=10,
                 device=torch.device('cpu')):
        # Parameters
        self.niters = niters
        self.nrows = n_superpixels_row
        self.ncols = n_superpixels_col
        self.nsup = self.nrows*self.ncols
        self.compactness = compactness
        self.stage1_nlayers = stage1_nlayers
        self.stage1_nconv = stage1_nconv
        self.device = device
        
        # Stabilization constant
        self.const = 1e2

        # Initialize parent
        super(SSN, self).__init__()
        
        # Create ReLU module
        self.relu = nn.ReLU(inplace=True)
        
        # Create first partial stage
        self.stage1_list = [nn.Conv2d(3, self.stage1_nconv, (3, 3), padding=(1, 1)),
                            self.relu]
        
        for idx in range(self.stage1_nconv-2):
            self.stage1_list.append(nn.Conv2d(self.stage1_nconv, self.stage1_nconv,
                                              3, padding=(1, 1)))
            self.stage1_list.append(self.relu)
            
        # Keep last stage without relu
        self.stage1_list.append(nn.Conv2d(self.stage1_nconv, self.stage1_nconv-5,
                                          (3, 3), padding=(1, 1)))
        
        # Now create stage1
        self.stage1 = nn.Sequential(*self.stage1_list)
        
        # create shifting convolution kernels
        self.shifter_size = self.stage1_nconv
        #self.shifters = nn.Conv2d(shifter_size, shifter_size, (3, 3),
        #                          padding=(1, 1), padding_mode='zeros',
        #                          groups=shifter_size, bias=False)
        #self.wshifters = nn.Conv2d(1, 1, (3, 3),
        #                          padding=(1, 1), padding_mode='zeros',
        #                          groups=1, bias=False)
        
        with torch.no_grad():
            self.shifter_window = Variable(torch.ones(self.shifter_size, 1, 3, 3,
                                                    device=self.device),
                                        requires_grad=False)
            self.wshifter_window = Variable(torch.ones(1, 1, 3, 3,
                                                        device=self.device),
                                            requires_grad=False)
            self.ckern_window = Variable(torch.ones(self.shifter_size, 1, 1, 1,
                                                    device=self.device),
                                        requires_grad=False)
            
            # Create kernel for compactness
            #self.ckern = nn.Conv2d(shifter_size, shifter_size, (1, 1),
            #                       groups=shifter_size, bias=False)
            
            #self.shifters.weight.requires_grad = False
            #self.wshifters.weight.requires_grad = False
            #self.ckern.weight.requires_grad = False

            #self.ckern.weight.fill_(1.0)
            #self.ckern.weight[[-2, -1], ...] = self.compactness
            self.ckern_window[[-2, -1], ...] = self.compactness
        
    def forward(self, X, **kwargs):
        '''
            Forward operator for genralized super pixel sampling
            
            Inputs:
                X: Input image of size B x 3 x H x W
                c2pix: HW x 9 Mapping between each pixel and it's 9 neighborhood
                    superpixels
                pix2c: 9-dimensional list of mapping between each centroid and
                    its neighbor superpixels
            
        '''
        _, _, H, W = X.shape
        
        # Modify compactness
        self.ckern_window[[-2, -1], ...] = self.compactness/np.sqrt(H*W/self.nsup)
        
        # Modify the constant
        self.const = 100
        
        # Generate meshgrid
        if X.is_cuda:
            device = X.get_device()
        else:
            device = torch.device('cpu')
            
        # For the image
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        
        Ygrid, Xgrid = torch.meshgrid(y, x)
        
        # For the superpixels
        yc = torch.arange(self.nrows, device=device)
        xc = torch.arange(self.ncols, device=device)
        
        # Upsample to directly sample the centroids
        Yc_lr, Xc_lr = torch.meshgrid(yc, xc)
        
        self.Xc = torch.nn.functional.interpolate(Xc_lr[None, None, ...].type(torch.float32),
                                                  size=(H, W),
                                                  mode='nearest')[0, 0, ...].type(torch.long)
        self.Yc = torch.nn.functional.interpolate(Yc_lr[None, None, ...].type(torch.float32),
                                                  size=(H, W),
                                                  mode='nearest')[0, 0, ...].type(torch.long)
        
        # Concatenate inputes to generate first five features
        imlabxy = torch.cat((X,
                             Xgrid[None, None, ...],
                             Ygrid[None, None, ...]), 1)
        
        # Generate first layer features
        imstage1 = self.stage1(X)
        
        # Concatenate the two to form features
        imfeat = torch.cat((imstage1, imlabxy), 1)
        #imfeat = imlabxy
        
        # Now perform iterations
        centroids, affinity = self.ssn_iter(imfeat)
        
        return centroids, affinity

    def ssn_iter(self, imfeat):
        '''
            Compute centroids and affinity maps
        '''
        B, nfeat, H, W = imfeat.shape
        
        # Get initial centroids by adaptive average pooling
        self.centroids = torch.nn.functional.adaptive_avg_pool2d(imfeat,
                                                                 (self.nrows,
                                                                  self.ncols))        

        # Create affinity matrix
        if imfeat.is_cuda:
            device = imfeat.device
        else:
            device = torch.device('cpu')
        
        # Compute affinity once
        self.compute_affinity(imfeat)
        
        # Now run iterations
        for _ in range(self.niters):
            self.compute_affinity(imfeat)
            self.compute_centroids(imfeat)
            
        return self.centroids, self.affinity
    
    def compute_affinity(self, imfeat):
        '''
            Compute B x 9 x HW soft affinity matrix
        ''' 
        _, _, H, W = imfeat.shape
        
        affinity_list = []
        
        idx = 0
        for r_shift in [-1, 0, 1]:
            Y1 = torch.clamp(self.Yc + r_shift, 0, self.nrows - 1)
            
            for c_shift in [-1, 0, 1]:
                X1 = torch.clamp(self.Xc + c_shift, 0, self.ncols-1)
                
                centroids_up = self.centroids[:, :, Y1, X1]
                conv_output = nn.functional.conv2d(imfeat - centroids_up,
                                                   self.ckern_window,
                                                   groups=self.shifter_size)
                affinity_list.append(-torch.pow(conv_output, 2).mean(1)[:, None, :, :])
                idx += 1
        
        # Compute softmax along feature dimension
        self.affinity = torch.cat(affinity_list, 1)
        self.affinity = (self.affinity/self.const).exp()
        
    def compute_centroids(self, imfeat):
        '''
            Computes centroids from affinity matrix
        '''
        B, nfeat, H, W = imfeat.shape
        
        # Zero out centroids
        self.centroids = self.centroids*0
        _, _, ch, cw = self.centroids.shape
        
        # Generate weights matrix too
        weights = self.centroids.data.new().resize_((B, nfeat, ch, cw)).fill_(0.0)
        weights = Variable(weights, requires_grad=True)
        
        idx = 0
        for r_shift in range(3):
            for c_shift in range(3):
                self.shifter_window[...] = 1
                #self.shifter_window[:, :, 2 - r_shift, 2 - c_shift] = 1
                
                #self.wshifter_window[...] = 0
                #self.wshifter_window[:, :, 2 - r_shift, 2 - c_shift] = 1
                
                affinity_sub = imfeat*self.affinity[:, [idx], :, :]
                centroid_sub = torch.nn.functional.adaptive_avg_pool2d(
                                                                affinity_sub,
                                                                (self.nrows,
                                                                 self.ncols))
                weight_sub = torch.nn.functional.adaptive_avg_pool2d(
                                                                self.affinity[:, [idx], :, :],
                                                                (self.nrows,
                                                                 self.ncols))
                
                # Shift and add centroids
                centroid_shifted = nn.functional.conv2d(centroid_sub,
                                                        self.shifter_window,
                                                        padding=1,
                                                        groups=self.shifter_size
                                                       )
                self.centroids = self.centroids + centroid_shifted
                
                # Shift and add weights
                weight_shifted = nn.functional.conv2d(weight_sub,
                                                      self.wshifter_window,
                                                      padding=1,
                                                      groups=1)
                
                weights = weights + weight_shifted
                idx = idx + 1
        
        self.centroids = self.centroids/weights
        
    @torch.no_grad()
    def get_hard_labels(self):
        '''
            Compute hard labels -- not differentiable
        '''
        label_img = torch.arange(self.nrows*self.ncols).reshape(self.nrows,
                                                                self.ncols)
        _, labels_sub = self.affinity[0, ...].max(0)
        H, W = labels_sub.shape
        
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        
        indices = torch.zeros(H, W, 9, dtype=torch.int16)
        idx = 0
        for r_idx in [-1, 0, 1]:
            Y1 = torch.clamp(self.Yc + r_idx, 0, self.nrows-1)
            for c_idx in [-1, 0, 1]:
                X1 = torch.clamp(self.Xc + c_idx, 0, self.ncols-1)                
                
                indices[:, :, idx] = label_img[Y1, X1]
                idx += 1
                
            
        labels = indices[Y, X, labels_sub]
        
        return labels
    
def create_mask(centroids, H, W, create_variable=False):
    '''
        Generate mask from centroid locations
        
        Inputs:
            centroids: 2 x ncentroids matrix with centroid locations
            H, W: Size of mask
            create_variable: If true, create an autograd variable
            
        Outputs:
            mask: 2D mask
    '''
    cx = torch.round(centroids[0, :]).type(torch.long)
    cy = torch.round(centroids[1, :]).type(torch.long)
    
    mask = torch.zeros(H, W, device=cx.device)
    
    if create_variable:
        mask = Variable(mask, requires_grad=True)
        mask.data[cy, cx] = 1
    else:
        mask[cy, cx] = 1
    
    return mask
                
@torch.no_grad()       
def rgb2lab(imrgb, normalize=False):
    '''
        Tensor implementation for converting rgb image to lab image. This code is
        a reimplementation of http://www.cs.tau.ac.il/~turkel/notes/RGB2Lab.m
        
        NOTE: The inputs are assumed to lie between [0, 1]
        
        Inputs:
            imrgb: B x 3 x H x W tensor
            normalize: If True, change values to lie between 0, 255
    '''    
    thres = 0.008856
    _, _, H, W = imrgb.shape
    
    imr = imrgb[:, [0], :, :]
    img = imrgb[:, [1], :, :]
    imb = imrgb[:, [2], :, :]
    
    # Convert to xyz
    imx = 0.412453*imr + 0.357580*img + 0.180423*imb
    imy = 0.212671*imr + 0.715160*img + 0.072169*imb
    imz = 0.019334*imr + 0.119193*img + 0.950227*imb
    
    # Normalize for D65 white point
    imx /= 0.950456
    imz /= 1.088754
    
    maskx = (imx > thres)
    masky = (imy > thres)
    maskz = (imz > thres)
    
    imy3 = torch.pow(imy, 1/3.0)
    
    fx = maskx*torch.pow(imx, 1/3.0) + (~maskx)*(7.787*imx + 16/116.0)
    fy = masky*imy3 + (~masky)*(7.787*imy + 16/116.0)
    fz = maskz*torch.pow(imz, 1/3.0) + (~maskz)*(7.787*imz + 16/116.0)
    
    L = masky*(116*imy3 - 16) + (~masky)*(903.3*imy)
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    
    imLab = torch.cat((L, a, b), 1)
    
    if normalize:
        imLab = (imLab - imLab.min())/(imLab.max() - imLab.min())
    
    return imLab

def rgb2lab_cu(imrgb):
    '''
        CuPy implementation for converting rgb image to lab image. This code is
        a reimplementation of http://www.cs.tau.ac.il/~turkel/notes/RGB2Lab.m
        
        NOTE: The inputs are assumed to lie between [0, 1]
        
        Inputs:
            imrgb: H x W x 3 cupy array
    '''    
    thres = 0.008856
    
    imr = imrgb[..., 0]
    img = imrgb[..., 1]
    imb = imrgb[..., 2]
    
    # Convert to xyz
    imx = 0.412453*imr + 0.357580*img + 0.180423*imb
    imy = 0.212671*imr + 0.715160*img + 0.072169*imb
    imz = 0.019334*imr + 0.119193*img + 0.950227*imb
    
    # Normalize for D65 white point
    imx /= 0.950456
    imz /= 1.088754
    
    maskx = (imx > thres)
    masky = (imy > thres)
    maskz = (imz > thres)
    
    imy3 = cupy.power(imy, 1/3.0)
    
    fx = maskx*cupy.power(imx, 1/3.0) + (~maskx)*(7.787*imx + 16/116.0)
    fy = masky*imy3 + (~masky)*(7.787*imy + 16/116.0)
    fz = maskz*cupy.power(imz, 1/3.0) + (~maskz)*(7.787*imz + 16/116.0)
    
    L = masky*(116*imy3 - 16) + (~masky)*(903.3*imy)
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    
    imLab = cupy.concatenate((L[..., cupy.newaxis],
                              a[..., cupy.newaxis],
                              b[..., cupy.newaxis]), axis=2)
    
    return imLab

def cu_soft_slic_efficient(imrgb, nrows, ncols, compactness=10.0, niters=10):
    '''
        Compute slic superpixels using SSN algorithm as stated in 
        https://varunjampani.github.io/papers/jampani18_SSN.pdf
        
        This function computes distances between each pixel and it's neighboring
        9 super pixels.
        
        Inputs:
            imrgb: H x W x 3 RGB image in cupy array format
            nrows: Number of superpixel rows
            ncols: Number of superpixel columns
            compactness: Weight for distance between coordinates
            niters: Number of iterations
            
        Outputs:
            centroids: nrows*ncols x 5 centroid matrix
            affinity: H*W x nrows*ncols affinity matrix
    '''
    H, W, _ = imrgb.shape
    const = 1000
    
    # Create rgb image
    S = cupy.sqrt(H*W/(nrows*ncols))
    C = compactness/S
    
    # Generate meshgrid
    X, Y = cupy.meshgrid(cupy.arange(W), cupy.arange(H))
    
    imlabxy = cupy.concatenate((imrgb,
                                X[..., cupy.newaxis],
                                Y[..., cupy.newaxis]), 2)
    
    # Generate initial centroids
    centroids = ndimage.zoom(imlabxy, (nrows/H, ncols/W, 1), order=1)
    
    # Reshape matrices
    centroids = centroids.reshape(nrows*ncols, 5)
    imlabxy = imlabxy.reshape(H*W, 5)
    
    # Pre-generate affinity matrix
    affinity = cupy.zeros((H, W, 9), dtype=cupy.float32)
    weights = cupy.zeros((nrows, ncols, 1), dtype=cupy.float32)
    c_weights = cupy.ones((1, 1, 5), dtype=cupy.float32)
    c_weights[:, :, 3:] = C
    
    Xm, Ym = cupy.meshgrid(cupy.arange(ncols), cupy.arange(nrows))
    Xup = ndimage.zoom(Xm, (H/nrows, W/ncols), order=0)
    Yup = ndimage.zoom(Ym, (H/nrows, W/ncols), order=0)
    
    index_base = cupy.arange(nrows*ncols).reshape(nrows, ncols)
    indices = cupy.zeros((H, W, 9), dtype=cupy.int16)
        
    # Now perform soft-slic iterations
    for _ in range(niters):
        # We will work in image mode
        centroids = centroids.reshape(nrows, ncols, 5)
        imlabxy = imlabxy.reshape(H, W, 5)
        
        # Step 1 -- compute affinity
        idx = 0
        for r_idx in [-1, 0, 1]:
            for c_idx in [-1, 0, 1]:
                X1 = cupy.clip(Xup + c_idx, 0, ncols-1)
                Y1 = cupy.clip(Yup + r_idx, 0, nrows-1)
                                
                indices[:, :, idx] = index_base[Y1, X1]
                
                c_img = centroids[Y1, X1, :]
                diff_img = -cupy.square(c_weights*(imlabxy - c_img)).mean(2)
                affinity[..., idx] = cupy.exp(diff_img/const)
                idx += 1
                            
        # Step 2 -- compute centroids
        centroids[...] = 0
        weights[...] = 0
        idx = 0
        for r_idx in [-1, 0, 1]:
            for c_idx in [-1, 0, 1]:
                c_img = imlabxy*affinity[..., [idx]]
                
                c_img_down = ndimage.zoom(c_img, (nrows/H, ncols/W, 1),
                                            order=1)
                w_img_down = ndimage.zoom(affinity[..., idx], (nrows/H, ncols/W),
                                            order=1)
                
                c_img_shift = cupy.roll(c_img_down, [r_idx, c_idx, 0], axis=[0, 1, 2])
                w_img_shift = cupy.roll(w_img_down, [r_idx, c_idx], axis=[0, 1])
                
                if r_idx == 1:
                    c_img_shift[0, :, :] = 0
                    w_img_shift[0, :] = 0
                elif r_idx == -1:
                    c_img_shift[-1, :, :] = 0
                    w_img_shift[-1, :] = 0
                    
                if c_idx == 1:
                    c_img_shift[:, 0, :] = 0
                    w_img_shift[:, 0] = 0
                elif c_idx == -1:
                    c_img_shift[:, -1, :] = 0
                    w_img_shift[:, -1] = 0
                
                centroids += c_img_shift
                weights += w_img_shift[:, :, cupy.newaxis]
                
                idx += 1

        weights[weights == 0] = 1
        centroids /= weights
        
    # Compute labels -- slightly difficult
    labels_sub = affinity.argmax(2)
    Y, X = cupy.mgrid[:H, :W]
    labels = indices[Y, X, labels_sub]
               
    return centroids, affinity, labels

def cu_soft_slic(imrgb, nrows, ncols, compactness=10.0, niters=10):
    '''
        Compute slic superpixels using SSN algorithm as stated in 
        https://varunjampani.github.io/papers/jampani18_SSN.pdf
        
        This function computes distances between all pixels and all superpixels
        
        Inputs:
            imrgb: H x W x 3 RGB image in cupy array format
            nrows: Number of superpixel rows
            ncols: Number of superpixel columns
            compactness: Weight for distance between coordinates
            niters: Number of iterations
            
        Outputs:
            centroids: nrows*ncols x 5 centroid matrix
            affinity: H*W x nrows*ncols affinity matrix
    '''
    H, W, _ = imrgb.shape
    const = 1000
    
    # Create rgb image
    imlab = rgb2lab_cu(imrgb)
    S = cupy.sqrt(H*W/(nrows*ncols))
    C = compactness/S
    
    # Generate meshgrid
    X, Y = cupy.meshgrid(cupy.arange(W), cupy.arange(H))
    
    imlabxy = cupy.concatenate((imlab, X[..., cupy.newaxis], Y[..., cupy.newaxis]), 2)
    
    # Generate initial centroids
    centroids = ndimage.zoom(imlabxy, (nrows/H, ncols/W, 1), order=1)
    
    # Reshape matrices
    centroids = centroids.reshape(nrows*ncols, 5)
    imlabxy = imlabxy.reshape(H*W, 5)
    
    # Pre-generate affinity matrix   
    affinity = cupy.zeros((H*W, nrows*ncols), dtype=cupy.float32)
    
    c_weights = cupy.ones((1, 5), dtype=cupy.float32)
    c_weights[:, 3:] = C
    
    # Now perform soft-slic iterations
    for _ in range(niters):
        
        # Step 1 -- compute affinity
        for c_idx in range(nrows*ncols):
            diff = -cupy.square(c_weights*(imlabxy - centroids[[c_idx], :])).mean(1)
            affinity[:, c_idx] = cupy.exp(diff/const)
        
        # Step 2 -- compute centroids
        centroids = cupy.dot(affinity.T, imlabxy)/affinity.sum(0).reshape(-1, 1)
        
    # Compute hard labels
    labels = affinity.argmax(1).reshape(H, W)
        
    return centroids.reshape(nrows, ncols, -1), affinity.reshape(H, W, -1), labels

    
if __name__ == '__main__':
    minsep = 31
    compactness = 10.0
    niters = 5
    expname = 'feathers_ms'
    
    plt.gray()
    #im = plt.imread('../../power_hs/imrgb.png')
    #im = plt.imread('../data/145059.jpg')
    #im = io.loadmat('../../power_hs/data/%s/%s.mat'%(expname, expname))['imrgb']
    im = io.loadmat('D:/Data/KAIST/slicedata_chunk_256/visitest.mat')['imrgb']
    im = im.astype(np.float32)/255
    
    H, W, _ = im.shape
    nrows = H // 10
    ncols = W // minsep
    
    imlab = color.rgb2lab(im).astype(np.float32)
    
    im_cu = cupy.array(im*255)
    im_ten = torch.tensor(im*255).permute(2, 0, 1)[None, :, :, :]
    
    net = SSN(niters, nrows, ncols, compactness)
    
    state_dict = torch.load('../models/rgbtest_5k.pth')
    net.load_state_dict(state_dict)
    
    # Send to cuda
    net = net.cuda()
    im_ten = im_ten.cuda()
    
    # Full -- 1.96s +/- 31.3ms
    # efficient -- 42ms
    # pytorch -- 44ms (good!)
    
    centroids1, affinity1 = net(im_ten)
    
    # Get hard labels
    labels1 = net.get_hard_labels()
    
    # Generate mask
    mask1 = create_mask(centroids1[0, -2:, :, :].reshape(2, -1), H, W).cpu().numpy()
    mask2 = cassi_cp.sanitize(mask1.astype(np.uint8), minsep).astype(np.float32)
    
    labels1_updated, N1 = cassi_cp.slic_update((im*255).astype(np.uint8),
                                               mask1.astype(np.uint8), compactness)
    
    centroids2, affinity2, labels2 = cu_soft_slic_efficient(im_cu, nrows, ncols, 5*compactness, niters)
    
    clr = (1, 1, 1)
    im_boundaries1 = segmentation.mark_boundaries(im, labels1.numpy(), color=clr)
    im_boundaries1_updated = segmentation.mark_boundaries(im, labels1_updated, color=clr)
    
    im_boundaries2 = segmentation.mark_boundaries(im, labels2.get(), color=clr)
    
    idx = 4
    
    plt.subplot(2, 2, 1)
    plt.imshow(im_boundaries1)
    plt.subplot(2, 2, 2)
    plt.imshow(im_boundaries2)
    
    mask_cat = np.zeros_like(im)
    mask_cat[..., 0] = mask1
    mask_cat[..., 1] = mask2
    plt.subplot(2, 2, 3)
    plt.imshow(mask_cat)
    plt.title('Before: %d; After: %d'%(mask1.sum(), mask2.sum()))
    
    plt.subplot(2, 2, 4)
    plt.imshow(im_boundaries1_updated)
    
    plt.show()