#!/usr/bin/env python

'''
     Module for generating datasets
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
from scipy.sparse import linalg as lin
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

class FlowDataset(torch.utils.data.Dataset):
    '''
        Data loader for loading images for pattern optimization

        Inputs:
            files_list: List of all files
            transform: (not in use)
            loadslice: If true, load a hyperspectral slice instead of cube
            masktype: 'random' or 'superpixel'

        Outputs:
            imrgb: RGB image
            mask: Random 2D sampling mask with constraints
    '''
    def __init__(self, files_list, transform=None, loadslice=True,
                 masktype='random'):
        '''
            Parse folder for data
        '''
        self.files_list = files_list
        self.loadslice = loadslice
        self.masktype = masktype
        
        # Load an image to find out image size
        matfile = io.loadmat(self.files_list[0])
        H, W, _ = matfile['imrgb'].shape
        self.imsize = [H, W]

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        matfile = io.loadmat(self.files_list[idx])

        # Extract images -- note that they are packed in uint8 form
        imrgb = matfile['imrgb'].astype(np.float32)
        H, W, _ = imrgb.shape
        
        try:
            minsep = matfile['wavelengths'].size
        except KeyError:
            minsep = 31
        
        if self.masktype == 'superpixel':
            # To speed up, the centroid masks and super pixels have been computed
            # before hand
            centroid_mask = matfile['centroid_mask'].astype(np.uint8)

            L2, N2 = cassi.cassi_cp.slic_update(imrgb, centroid_mask, 10.0)

            # Randomly intensify -- this should add some diversity to epochs
            mask = cassi.cassi_cp.random_intensify(centroid_mask, minsep)
            mask = mask.astype(np.float32)
        else:
            mask = cassi.get_random_mask(H, W, minsep).astype(np.float32)
            
        imrgb = torch.tensor(imrgb).permute([2, 0, 1])
        mask = torch.tensor(mask)
        
        return imrgb, mask           
        
class PanCNNSPDataset(torch.utils.data.Dataset):
    '''
        Data loader for sharpening hyperspectral images created using super
        pixel propagation

        Inputs:
            files_list: List of mat files to iterate over
            transform: (Not implemented)
            loadslice: If True, load HSI slices instead of cubes
            augment: If True, implement augmentation by random flipping,
            random_mask: If True, generate random mask. Else generate mask with
                superpixel centroids.
            getmask: If True, get the sampling mask
            reconstruct: If True, reconstruct the HSI band using super pixel
                propagation. Else, return the masked HSI band
            add_noise: Add readout + photon noise if true
            tau_max: Maximum photon level for noise simulation
            noise_snr: Readout noise (in electrons) for noise simulation

        Outputs:
            impan: Panchromatic (grayscale) image
            hsi_lr: Low resolution hyperspectral image (super pixel prop.)
            hsi_gt: Ground truth hyperspectral image
            mask: 2D sampling mask
    '''
    def __init__(self, files_list, transform=None, loadslice=False, augment=False,
                 random_mask=False, getmask=False, reconstruct=True,
                 add_noise=True, tau_max=2000, noise_snr=5,
                 patch_size=(128, 128), minsep=31):
        '''
            Parse folder for data
        '''
        self.files_list = files_list
        self.loadslice = loadslice
        self.random_mask = random_mask
        self.getmask = getmask
        self.add_noise = add_noise
        self.tau_max = tau_max
        self.noise_snr = noise_snr
        self.augment = augment
        self.reconstruct = reconstruct
        self.patch_size = patch_size
        self.minsep = minsep
        
        # Possible scales
        self.scales = np.array([1, 2, 3, 4])

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        matfile = io.loadmat(self.files_list[idx])

        # Extract images -- note that they are packed in uint8 form
        imrgb = matfile['imrgb']
        if self.loadslice:
            hypercube = matfile['hyperimg'].astype(np.float32)/255
            hypercube = hypercube[:, :, np.newaxis]
        else:
            hypercube = matfile['hypercube'].astype(np.float32)/255

        # To speed up, the centroid masks and super pixels have been computed
        # before hand
        centroid_mask = matfile['centroid_mask'].astype(np.uint8)
        H, W = centroid_mask.shape
        
        if self.augment:
            # Random crop
            start_h = np.random.randint(0, H - self.patch_size[0])
            start_w = np.random.randint(0, W - self.patch_size[1])
                        
            imrgb = imrgb[start_h:start_h+self.patch_size[0],
                          start_w:start_w+self.patch_size[1], :]
            hypercube = hypercube[start_h:start_h+self.patch_size[0],
                                  start_w:start_w+self.patch_size[1], :]
            centroid_mask = centroid_mask[start_h:start_h+self.patch_size[0],
                                          start_w:start_w+self.patch_size[1]]
            # Randomly horizontal flip
            if np.random.rand() < 0.5:
                imrgb = np.copy(imrgb[:, ::-1, :], order='C')
                hypercube = np.copy(hypercube[:, ::-1, :], order='C')
                centroid_mask = np.copy(centroid_mask[..., ::-1], order='C')
            # Random vertical flip
            if np.random.rand() < 0.5:
                imrgb = np.copy(imrgb[::-1, :, :], order='C')
                hypercube = np.copy(hypercube[::-1, ...], 'C')
                centroid_mask = np.copy(centroid_mask[::-1, ...], 'C')

        # Convert RGB image to Panchromatic image
        # NOTE: squaring the imrgb image and then adding creates better results

        #impan = ((imrgb.astype(np.float32)/255)**2).mean(2)
        impan = imrgb.astype(np.float32).mean(2)/255

        if self.random_mask:
            H, W = impan.shape
            mask = cassi.get_random_mask(H, W, self.minsep).astype(np.float32)
            
            L2 = None
            N2 = mask.sum()
        else:
            L2, N2 = cassi.cassi_cp.slic_update(imrgb, centroid_mask, 10.0)

            # Randomly intensify -- this should add some diversity to epochs
            mask = cassi.cassi_cp.random_intensify(centroid_mask, self.minsep)
            mask = mask.astype(np.float32)

        # Now reconstruct
        if self.add_noise:
            self.tau = 10 + int(np.random.rand()*(self.tau_max-10))
            img = utils.measure(hypercube, self.noise_snr, self.tau)*mask[:, :, np.newaxis]
        else:
            img = hypercube*mask[..., np.newaxis]
            
        if self.reconstruct:        
            hsi_lr = cassi.cassi_cp.recon_superpixel(img.astype(np.float32),
                                                    mask,
                                                    L2, impan, int(N2))
        else:
            hsi_lr = img.astype(np.float32)
    
        # Convert to tensor, and shuffle the dimensions so that lambda is first
        # axis
        impan = torch.tensor(impan)
        hypercube = torch.tensor(hypercube).permute([2, 0, 1])
        hsi_lr = torch.tensor(hsi_lr).permute([2, 0, 1])
        mask = torch.tensor(mask.astype(np.float32))

        # Convert data to tensor
        if self.getmask:
            return impan, hsi_lr, hypercube, mask
        else:
            return impan, hsi_lr, hypercube

class PanCNNLRDataset(torch.utils.data.Dataset):
    '''
        Data loader for sharpening hyperspectral images created using super
        pixel propagation, and low-rank demomposed

        Inputs:
            files_list: List of mat files to iterate over
            rank: Rank for decomposing the HSI
            getmask: If True, get the sampling mask
            reconstruct: If True, reconstruct the HSI band using super pixel
                propagation. Else, return the masked HSI band
            add_noise: Add readout + photon noise if true
            tau_max: Maximum photon level for noise simulation
            noise_snr: Readout noise (in electrons) for noise simulation

        Outputs:
            impan: Panchromatic (grayscale) image
            hsi_lr: Low resolution hyperspectral image (super pixel prop.)
            hsi_gt: Ground truth hyperspectral image
            mask: 2D sampling mask
    '''
    def __init__(self, files_list, rank=6, getmask=False, reconstruct=True,
                 add_noise=True, tau_max=2000, noise_snr=5, minsep=31):
        '''
            Parse folder for data
        '''
        self.files_list = files_list
        self.rank = rank
        self.getmask = getmask
        self.add_noise = add_noise
        self.tau_max = tau_max
        self.noise_snr = noise_snr
        self.reconstruct = reconstruct
        self.minsep = minsep
        
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        matfile = io.loadmat(self.files_list[idx])

        # Extract images -- note that they are packed in uint8 form
        imrgb = matfile['imrgb']
        hypercube = matfile['hypercube'].astype(np.float32)/255

        # To speed up, the centroid masks and super pixels have been computed
        # before hand
        centroid_mask = matfile['centroid_mask'].astype(np.uint8)
        H, W = centroid_mask.shape
        
        # Convert RGB image to Panchromatic image
        # NOTE: squaring the imrgb image and then adding creates better results

        #impan = ((imrgb.astype(np.float32)/255)**2).mean(2)
        impan = imrgb.astype(np.float32).mean(2)/255

        L2, N2 = cassi.cassi_cp.slic_update(imrgb, centroid_mask, 10.0)

        # Randomly intensify -- this should add some diversity to epochs
        mask = cassi.cassi_cp.random_intensify(centroid_mask, self.minsep)
        mask = mask.astype(np.float32)

        # Now reconstruct
        if self.add_noise:
            #self.tau = 10 + int(np.random.rand()*(self.tau_max-10))
            self.tau = self.tau_max
            img = utils.measure(hypercube, self.noise_snr, self.tau)*mask[:, :, np.newaxis]
        else:
            img = hypercube*mask[..., np.newaxis]
            
        if self.reconstruct:        
            hsi_lr = cassi.cassi_cp.recon_superpixel(img.astype(np.float32),
                                                    mask,
                                                    L2, impan, int(N2))
        else:
            hsi_lr = img.astype(np.float32)
            
        # Decompose the hyperspectral image
        H, W, T = hsi_lr.shape
        hsmat = hsi_lr.reshape(H*W, T)
        u, s, vt = lin.svds(hsmat, self.rank)
        indices = np.argsort(s)[::-1]
 
        u = u[:, indices]
        s = s[indices]
        vt = vt[indices, :]
 
        u_img = u.reshape(H, W, self.rank)

        # Also decompose the groundtruth HSI -- we will try reconstructing the
        # spatial singular vectors alone
        ugt, sgt, _ = lin.svds(hypercube.reshape(H*W, T), self.rank)
        indices = np.argsort(sgt)[::-1]

        ugt = ugt[:, indices].reshape(H, W, self.rank)
    
        # Convert to tensor, and shuffle the dimensions so that lambda is first
        # axis
        impan = torch.tensor(impan)
        hypercube = torch.tensor(hypercube).permute([2, 0, 1])
        u_img = torch.tensor(u_img).permute(2, 0, 1)
        ugt = torch.tensor(ugt).permute(2, 0, 1)
        S = torch.tensor(np.diag(s))
        vt = torch.tensor(vt)
        mask = torch.tensor(mask.astype(np.float32))

        return impan, u_img, S, vt, hypercube, mask, ugt

class PanCNNRealDataset(torch.utils.data.Dataset):
    '''
        Data loader for real HSI data

        Inputs:
            files_list: List of mat files to iterate over
            reconstruct: If True, reconstruct the HSI band using super pixel
                propagation. Else, return the masked HSI band
            add_noise: Add readout + photon noise if true
            tau_max: Maximum photon level for noise simulation
            noise_snr: Readout noise (in electrons) for noise simulation

        Outputs:
            impan: Panchromatic (grayscale) image
            hsi_lr: Low resolution hyperspectral image (super pixel prop.)
            hsi_gt: Ground truth hyperspectral image
            mask: 2D sampling mask
    '''
    def __init__(self, files_list, reconstruct=True, loadslice=False,
                 add_noise=True, tau_max=2000, noise_snr=5):
        '''
            Parse folder for data
        '''
        self.files_list = files_list
        self.reconstruct = reconstruct
        self.loadslice = loadslice
        self.add_noise = add_noise
        self.tau_max = tau_max
        self.noise_snr = noise_snr
        
        self.compactness = 10.0
        
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        matfile = io.loadmat(self.files_list[idx])
        
        # Load images 
        imrgb = matfile['guide_img']
        mask = matfile['mask']
        
        impan = imrgb.astype(np.float32).mean(2)/255
        
        if self.loadslice:
            hypercube = matfile['hyperimg'].astype(np.float32)[:, :, np.newaxis]
            sassicube = matfile['sassi_img'].astype(np.float32)[:, :, np.newaxis]
        else:
            hypercube = matfile['hypercube'].astype(np.float32)
            sassicube = matfile['sassi_cube'].astype(np.float32)
        
        # Normalize values in both cubes    
        maxval = hypercube.max()
        
        hypercube /= maxval
        sassicube /= maxval
        
        # Get membership
        L2, N2 = cassi.slic_update(imrgb, mask, self.compactness)
        
        # Now reconstruct
        if self.add_noise:
            self.tau = 10 + int(np.random.rand()*(self.tau_max-10))
            img = utils.measure(sassicube, self.noise_snr, self.tau)*mask[:, :, np.newaxis]
        else:
            img = sassicube*mask[..., np.newaxis]
            
        if self.reconstruct:        
            hsi_lr = cassi.cassi_cp.recon_superpixel(img.astype(np.float32),
                                                    mask.astype(np.float32),
                                                    L2, impan, int(N2))
            hsi_lr = np.clip(hsi_lr, 0, 1)
        else:
            hsi_lr = img.astype(np.float32)
            
        # Convert to tensor, and shuffle the dimensions so that lambda is first
        # axis
        impan = torch.tensor(impan)
        hypercube = torch.tensor(hypercube).permute([2, 0, 1])
        hsi_lr = torch.tensor(hsi_lr).permute([2, 0, 1])
        mask = torch.tensor(mask.astype(np.float32))

        # Convert data to tensor
        return impan, hsi_lr, hypercube, mask
class PanCNNDataset(torch.utils.data.Dataset):
    '''
        Data loader for training Panchromatic sharpening using CNNs

        Inputs:
            root: Folder where all data is stored
            bayer_data: Spectral response of an RGB camera to get Panchromatic
                image
            scaling: Resizing factor to simulate downsampling operator
            subsample: If True, subsample instead of downsampling

        Outputs:
            Decided by torch
    '''
    def __init__(self, files_list, scaling=4, subsample=False, augment=False,
                 transform=None, loadslice=False, getmask=False):
        '''
            Parse folder for data
        '''
        self.files_list = files_list
        self.scaling = scaling
        self.subsample = subsample
        self.loadslice = loadslice
        self.getmask = getmask
        self.augment = augment

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        matfile = io.loadmat(self.files_list[idx])

        # Extract images -- note that they are packed in uint8 form
        imrgb = matfile['imrgb'].astype(np.float32)/255
        if self.loadslice:
            hypercube = matfile['hyperimg'].astype(np.float32)/255
            hypercube = hypercube[:, :, np.newaxis]
            
            try:
                minsep = matfile['wavelengths'].size
            except KeyError:
                minsep = 31
                
        else:
            hypercube = matfile['hypercube'].astype(np.float32)/255
            minsep = hypercube.shape[2]
            
        # To speed up, the centroid masks and super pixels have been computed
        # before hand
        centroid_mask = matfile['centroid_mask'].astype(np.uint8)
            
        if self.augment:
            # Randomly horizontal flip
            if np.random.rand() < 0.5:
                imrgb = np.copy(imrgb[:, ::-1, :], order='C')
                hypercube = np.copy(hypercube[:, ::-1, :], order='C')
                centroid_mask = np.copy(centroid_mask[..., ::-1], order='C')
            # Random vertical flip
            if np.random.rand() < 0.5:
                imrgb = np.copy(imrgb[::-1, :, :], order='C')
                hypercube = np.copy(hypercube[::-1, ...], 'C')
                centroid_mask = np.copy(centroid_mask[::-1, ...], 'C')

        # Convert RGB image to Panchromatic image
        impan = imrgb.astype(np.float32).mean(2)/255

        L2, N2 = cassi.cassi_cp.slic_update(imrgb, centroid_mask, 10.0)

        # Randomly intensify -- this should add some diversity to epochs
        mask = cassi.cassi_cp.random_intensify(centroid_mask, minsep)
        mask = mask.astype(np.float32)

        # Create downsampled hypercube
        if self.subsample:
            hsi_lr = hypercube[::self.scaling, ::self.scaling, :]
        else:
            ratio = 1.0/self.scaling
            hsi_lr = cv2.resize(hypercube, None, fx=ratio, fy=ratio)

        if self.loadslice:
            hsi_lr = hsi_lr[:, :, None]

        # Convert to tensor, and shuffle the dimensions so that lambda is first
        # axis
        impan = torch.tensor(impan)
        hypercube = torch.tensor(hypercube).permute([2, 0, 1])
        hsi_lr = torch.tensor(hsi_lr).permute([2, 0, 1])
        mask = torch.tensor(mask)

        # Convert data to tensor
        if self.getmask:
            return impan, hsi_lr, hypercube, mask
        else:
            return impan, hsi_lr, hypercube
