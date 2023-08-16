#!/usr/bin/env python

'''
Description of files goes here.
'''

# System imports
import os
import sys
import pdb
from cupy import is_available

# Scientific computing
import numpy as np
from scipy import signal
from scipy import io
from scipy.sparse import linalg
from scipy import interpolate

import cv2
import matplotlib.pyplot as plt
from skimage import segmentation
from sklearn.utils.extmath import randomized_svd

import torch

try:
    from fast_slic.avx2 import SlicAvx2
except ImportError:
    from fast_slic import Slic as SlicAvx2

from modules import hyperspectral
from modules import utils
from modules import cassi_cp
from modules import filtering

def hsi_lr_projection(hsi, rank=6):
    '''
        Perform a low-rank decomposition on the hyperspectral image
        
        Inputs:
            hsi: HxWxT hyperspectral image
            rank: Rank to decompose the HSI
            
        Outputs:
            hsi_lr: Low rank approximation of HSI
    '''
    H, W, T = hsi.shape
    
    hsi_mat = hsi.reshape(H*W, T)
    
    U, S, Vt = randomized_svd(hsi_mat,
                              n_components=rank,
                              n_iter=2*rank,
                              random_state=None)
    hsi_mat_lr = U.dot(np.diag(S)).dot(Vt)
    
    return hsi_mat_lr.reshape(H, W, T)

def recon_sparse_pancnn(hsi_inflated, impan, mask, net):
    '''
        Wrapper function to reconstruct full HSI from sparsely sampled HSI using
        neural networks.

        Inputs:
            hsi_inflated: HxWxT 3D hyperspectral tensor with non-zero values
                only at the sampled locations
            impan: Grayscale image of the scene
            mask: 2D mask used for created hsi_inflated
            net: PanCNNGuided trained network.

        Outputs:
            hsi_recon: Reconstructed hyperspectral image

    '''
    H, W, T = hsi_inflated.shape
    hsi_recon = np.zeros((H, W, T), dtype=np.float32)

    hsi_cat = torch.zeros((1, 3, H, W))
    hsi_cat[0, 1, :, :] = torch.tensor(impan)
    hsi_cat[0, 2, :, :] = torch.tensor(mask)

    for idx in range(T):
        # Reconstruction is per channel
        hsi_cat[0, 0, :, :] = torch.tensor(hsi_inflated[:, :, idx])

        # Reconstruct the band
        with torch.no_grad():
            hsi_rec = net(hsi_cat)
            _, _, H2, W2 = hsi_rec.shape

        if H > H2:
            dH = (H - H2)//2
            dW = (W - W2)//2
            hsi_recon[dH:-dH, dW:-dW, idx] = hsi_rec[0, 0, :, :].detach().numpy()
        else:
            # And reassign
            hsi_recon[:, :, idx] = hsi_rec[0, 0, :, :].detach().numpy()

    hsi_recon[hsi_recon < 0] = 0
    hsi_recon[hsi_recon > 1] = 1

    return hsi_recon

def get_random_mask(H, W, minsep=31):
    '''
        Function to obtain random sampling mask with minimum separation
        
        Inputs:
            H, W: Size of the mask
            minsep: Minimum separation between two openings
            
        Outputs:
            mask: Random sampling mask
    '''
    # Generate
    mask = (np.random.rand(H, W) < 1/minsep).astype(np.uint8)
    
    # Sanitize
    mask = cassi_cp.sanitize(mask, minsep)
    
    # Intensify
    mask = cassi_cp.random_intensify(mask, minsep)
    
    return mask    

def sanitize(mask, minsep):
    '''
        Function to sanitize a CASSI mask such that no two openings along a 
        row are separated by less than minsep

        Inputs:
            mask: Binary mask with 1 at spatial location having openings
            minsep: Minimum separation between two openings along the row

        Outputs:
            mask: Sanitized CASSI mask
    '''
    H, W = mask.shape

    for h in range(H):
        next_idx = 0

        while next_idx < W-1:
            if mask[h, next_idx] == 1:
                next_idx2 = min(W, next_idx + minsep)
                if h < H-1:
                    mask[h+1, next_idx+1:next_idx2] = (
                        mask[h, next_idx+1:next_idx2])

                mask[h, next_idx+1:next_idx2] = 0
                next_idx += minsep
            else:
                next_idx += 1

    return mask

def intensify(mask, minsep):
    '''
        Function to increase density of openings while ensuring that openings
        are separated by minsep.

        Inputs:
            mask: Binary mask
            minsep: Minimum separation

    '''
    H, W = mask.shape

    # Create dilated image to estimate plausible locations
    kern = np.ones((1, 2*minsep-1), dtype=np.uint8)
    mask_conv = cv2.dilate(mask, kern, 1)

    for h in range(H):
        next_idx = 0

        while next_idx < W:
            if mask[h, next_idx] == 1:
                next_idx += minsep
            else:
                if mask_conv[h, next_idx] == 0:
                    mask[h, next_idx] = 1
                    next_idx += minsep
                else:
                    next_idx += 1

    return mask

def random_intensify(mask, minsep):
    '''
        Randomly intensify a sparse CASSI mask

        Inputs:
            mask: Sparse CASSI mask which may have fewer than critical number
                of openings
            minsep: Minimum separation between two openings

        Outputs:
            mask_intense: Mask with more openings
    '''
    H, W = mask.shape
    [mh, mw] = np.where(mask == 1)

    # First create a random mask
    random_mask = (np.random.rand(H, W) < 2/minsep).astype(np.float32)

    # Sanitize the random mask
    random_mask = sanitize(random_mask, minsep)

    # Add the two masks together, but then sanitize once more around the 
    # input mask locations
    random_mask[mh, mw] = 0

    mask_intense = random_mask + mask

    # Now do additional sanitization
    for idx in range(mh.size):
        h = mh[idx]
        w = mw[idx]

        mask_intense[h, max(w-minsep+1, 0):w] = 0
        mask_intense[h, (w+1):min(w+minsep, W)] = 0

    return mask_intense

def get_sp_mask(imrgb, minsep, frac=0.25, compactness=10):
    '''
        Function to get CASSI sampling mask using centroids of super pixels.

        Inputs:
            imrgb: RGB image
            minsep: Minimum separation between openings
            frac (0.25): Fraction of H*W/minsep for number of super pixels
            compactness (10): Compactness parameter for SLIC

        Outputs:
            centroid_mask: Mask with centroids generated from super pixelation
            mask: Sampling mask
    '''

    H, W, _ = imrgb.shape

    # Create slic module
    n_components = int(frac*H*W/minsep)
    slic = SlicAvx2(num_components=n_components, compactness=compactness)

    # Create Lab image
    imLab = cv2.cvtColor(imrgb, cv2.COLOR_RGB2Lab)

    # Generate superpixels
    superpixels = slic.iterate(imLab)

    centers = np.array([cluster['yx'] for cluster in slic.slic_model.clusters])

    # Create centroid mask and sanitize
    centroid_mask = np.zeros((H, W), dtype=np.uint8)
    centroid_mask[centers[:, 0].astype(int), centers[:, 1].astype(int)] = 1

    centroid_mask = cassi_cp.sanitize(centroid_mask, minsep)

    # Then intensify
    mask = cassi_cp.random_intensify(centroid_mask, minsep)

    return centroid_mask, mask

@torch.no_grad()
def recon_guided(hsi_inflated, impan, mask, winsize=[31, 31], lr_filter=False,
                 rank=10, wintype='gaussian'):
    '''
        Reconstruct a hyperspectral image from sparse measurements using
        guided filter.

        Inputs:
            hsi_inflated: HxWxT hyperspectral image with non-zeros at sampled
                locations
            impan: HxW panchromatic guide image
            mask: HxW sampling mask
            winsize: Size of window for reconstruction
            lr_filter: If True, perform a low-rank projection
            rank: Rank for low-rank filtering
            wintype: 'ones' or 'gaussian'

        Outputs:
            hsi_recon: Reconstructed HSI
            
        NOTE: This code relies on torch+CUDA
    '''
    eps = 1e-6
    H, W, T = hsi_inflated.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    conv_kernel = torch.zeros(1, 1, winsize[0], winsize[1], device=device)
    
    if wintype == 'gaussian':
        window = filtering.gausswin2d(winsize)
    else:
        window = np.ones(winsize)
        
    conv_kernel[0, 0, ...] = torch.tensor(window, device=device)
    padding = [winsize[0]//2, winsize[1]//2]
    boxfilter = torch.nn.Conv2d(1, 1, winsize, padding=padding,
                                   padding_mode='reflect', bias=False).to(device)
    boxfilter.weight.requires_grad = False
    boxfilter.weight[...] = conv_kernel
    
    hsi_tensor = torch.tensor(hsi_inflated.astype(np.float32).transpose(2, 0, 1)[:, np.newaxis, :, :],
                              device=device)
    
    mask_tensor = torch.tensor(mask.astype(np.float32), device=device)
    mask_tensor = torch.repeat_interleave(mask_tensor[None, None, ...], T, 0)
    
    guide_tensor = torch.tensor(impan.astype(np.float32), device=device)
    guide_tensor = torch.repeat_interleave(guide_tensor[None, None, ...], T, 0)
    
    inputs = torch.cat((hsi_tensor, guide_tensor, mask_tensor), 1)
    
    outputs = filtering.guided_filter(inputs, boxfilter, True, eps)
    
    hsi_recon = outputs.detach().cpu()[:, 0, :, :].numpy().transpose(1, 2, 0)
    
    return hsi_recon

def recon_guided_old(hsi_inflated, impan, mask, winsize=[31, 31], lr_filter=False,
                 rank=10, wintype='gaussian'):
    hsi_recon = np.zeros_like(hsi_inflated)

    T = hsi_inflated.shape[2]
    winsize = [winsize[0], 2*T+1]

    for idx in range(hsi_inflated.shape[2]):
        hsi_recon[:, :, idx] = filtering.guidedFilterMasked(impan,
                                                            hsi_inflated[..., idx],
                                                            mask, winsize[0], eps)
        
    if lr_filter:
        hsi_recon = hsi_lr_projection(hsi_recon, rank=rank)

    return hsi_recon

def slic_update(imrgb, mask, compactness=10.0):
    '''
        Performs one single step of SLIC to update membership

        Inputs:
            imrgb: RGB image
            mask: Sparse sampling mask / centroids of superpixels after
                sanitizing
            compactness: SLIC compactness parameter

        Outputs:
            L: superpixel membership map
            N: Total number of super pixels
    '''

    H, W, _ = imrgb.shape
    ch, cw = np.where(mask == 1)

    # Create LabXY image
    [Y, X] = np.mgrid[:H, :W]

    imlabxy = np.zeros((H, W, 5), dtype=np.float32)

    imlabxy[:, :, :3] = cv2.cvtColor(imrgb, cv2.COLOR_RGB2Lab)
    imlabxy[:, :, 3] = X
    imlabxy[:, :, 4] = Y

    # Reshape to a matrix
    imlabxymat = imlabxy.reshape(H*W, 5).astype(np.float32)

    centroids_labxy = imlabxy[ch, cw, :].astype(np.float32)
    N = ch.size
    S = int(np.sqrt(H*W/N))

    nmembers = np.zeros(N)

    dist_matrix = np.ones((H, W), dtype=np.float32)*float('inf')
    L = np.ones((H, W), dtype=np.uint16)

    # Inefficient, but just do it
    for idx in range(N):
        hmin = max(0, ch[idx] - 2*S); hmax = min(H, ch[idx] + 2*S)
        wmin = max(0, cw[idx] - 2*S); wmax = min(W, cw[idx] + 2*S)

        imlabxy_patch = imlabxy[hmin:hmax, wmin:wmax, :]
        dist_patch_old = dist_matrix[hmin:hmax, wmin:wmax]
        dist_patch = cassi_cp._get_dist_cp(centroids_labxy[idx, :], imlabxy_patch,
                                           np.float32(compactness), S)

        L_patch = L[hmin:hmax, wmin:wmax]
        L_patch[dist_patch < dist_patch_old] = idx

        L[hmin:hmax, wmin:wmax] = L_patch

        dist_matrix[hmin:hmax, wmin:wmax] = np.minimum(dist_patch,
                                                       dist_patch_old)

    return L.astype(np.uint16), N

def _get_dist(centroid_labxy, imlabxy_patch, compactness, S):
    '''
        Internal function to compute distance from all centroids
    '''
    h, w, _ = imlabxy_patch.shape
    imlabxy_mat = imlabxy_patch.reshape(h*w, -1)

    diff_mat = (imlabxy_mat - centroid_labxy)**2
    dist1 = diff_mat[:, :3].sum(1)
    dist2 = diff_mat[:, 3:].sum(1)

    dist = np.sqrt(dist1 + (compactness/S)*dist2).reshape(h, w)

    return dist

@torch.no_grad()
def recon_superpixel_lr(hsi_inflated, mask, labels, impan, N, net=None,
                        lr_filter=True, rank=6, recon_3d=False):
    '''
        Reconstruct the hyperspectral image using superpixel propagation
        approach and then reconstruction of spatial singular vectors using
        neural network (if provided)

        Inputs:
            hsi_inflated: HxWxT cube with spectrum only at sampled locations
            mask: HxW sampling mask
            labels: HxW image with membership information
            impan: Grayscale guide/panchromatic image
            N: Number of superpixels
            net: If PanCNN network is provided, then the propagated HSI is
                further filtered using the network.
            lr_filter: If True, perform a low-rank projection
            rank: Rank for low-rank filtering
            recon_3d: If True, use PanCNN3DGuided architecture

        Outputs:
            hsi_rec: Reconstructed hsi
    '''
    H, W, T = hsi_inflated.shape

    # First reconstruct the full HSI
    hsi_rec = cassi_cp.recon_superpixel(hsi_inflated, mask,
                                        labels, impan, N)
    
    # Clamp results to [0, 1]
    hsi_rec = np.clip(hsi_rec, 0, 1)
    
    if lr_filter:
        u, s, vt = linalg.svds(hsi_rec.reshape(H*W, T), rank)
        u_img = u.reshape(H, W, rank)
        
        # Now check if net is provided
        if net is not None:
            if next(net.parameters()).is_cuda:
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            if recon_3d:
                u_tensor = torch.tensor(u_img).permute(2, 0, 1)[None, ...]
                inputs = torch.cat((u_tensor,
                                torch.tensor(impan)[None, None, ...]), 1)
                inputs = inputs.to(device)
                u_img = net(inputs).detach().cpu().numpy()[0, ...].transpose(1, 2, 0)
                
            else:
                if net.maskmode:
                    inputs = torch.zeros((1, 3, H, W))
                    inputs[0, 2, :, :] = torch.tensor(mask)
                else:
                    inputs = torch.zeros((1, 2, H, W))
                inputs[0, 1, :, :] = torch.tensor(impan)
                
                for idx in range(rank):
                    if net.maskmode:
                        inputs[0, 0, :, :] = torch.tensor(u_img[:, :, idx]*mask)
                    else:
                        inputs[0, 0, :, :] = torch.tensor(u_img[:, :, idx])
                    inputs = inputs.to(device)
                    u_rec = net(inputs).detach().cpu().numpy()
                    
                    u_img[..., idx] = u_rec[0, 0, ...]
            
        hsi_rec = (u_img.reshape(H*W, -1)).dot(np.diag(s)).dot(vt).reshape(H, W, -1)
        
    return hsi_rec

@torch.no_grad()
def recon_superpixel(hsi_inflated, mask, labels, impan, N, renorm=True, net=None,
                     lr_filter=False, rank=10, recon_3d=False):
    '''
        Reconstruct the hyperspectral image using superpixel propagation
        approach

        Inputs:
            hsi_inflated: HxWxT cube with spectrum only at sampled locations
            mask: HxW sampling mask
            labels: HxW image with membership information
            impan: Grayscale guide/panchromatic image
            N: Number of superpixels
            renorm: If True, renormalize reconstruction so that the gray scale
                image looks like impan
            net: If PanCNN network is provided, then the propagated HSI is
                further filtered using the network.
            lr_filter: If True, perform a low-rank projection
            rank: Rank for low-rank filtering
            recon_3d: If True, use PanCNN3DGuided architecture

        Outputs:
            hsi_rec: Reconstructed hsi
            
    '''
    H, W, T = hsi_inflated.shape

    if (net is None) or ( (net is not None) and (net.maskmode is False)):
        # First reconstruct the full HSI
        hsi_rec = cassi_cp.recon_superpixel(hsi_inflated, mask,
                                            labels, impan, N)
        
        # Clamp results to [0, 1]
        hsi_rec = np.clip(hsi_rec, 0, 1)

    else:
        hsi_rec = np.zeros_like(hsi_inflated)

    # Now check if net is provided
    if net is not None:
        if next(net.parameters()).is_cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if recon_3d:
            if net.maskmode:
                hsi_tensor = torch.tensor(hsi_inflated).permute(2, 0, 1)[None, ...]
                inputs = torch.cat((hsi_tensor,
                                    torch.tensor(impan)[None, None, ...],
                                   torch.tensor(mask)[None, None, ...]), 1)
            else:
                hsi_tensor = torch.tensor(hsi_rec).permute(2, 0, 1)[None, ...]
                inputs = torch.cat((hsi_tensor,
                                    torch.tensor(impan)[None, None, ...]), 1)
            inputs = inputs.to(device)
            hsi_rec = net(inputs).detach().cpu().numpy()[0, ...].transpose(1, 2, 0)
            
        else:
            if net.maskmode:
                inputs = torch.zeros((1, 3, H, W))
                inputs[0, 2, :, :] = torch.tensor(mask)
            else:
                inputs = torch.zeros((1, 2, H, W))
            inputs[0, 1, :, :] = torch.tensor(impan)
            
            for idx in range(T):
                if net.maskmode:
                    inputs[0, 0, :, :] = torch.tensor(hsi_inflated[:, :, idx])
                else:
                    inputs[0, 0, :, :] = torch.tensor(hsi_rec[:, :, idx])
                inputs = inputs.to(device)
                outputs = net(inputs).detach().cpu().numpy()

                hsi_rec[:, :, idx] = outputs[0, 0, :, :]
                
    if lr_filter:
        hsi_rec = hsi_lr_projection(hsi_rec, rank=rank)
        
    # Check if renormalization required
    if renorm:
        hsi_gray = hsi_rec.mean(2)
        imratio = (impan/(hsi_gray + 1e-2*hsi_gray.max()))
        hsi_rec *= imratio[..., np.newaxis]

    return hsi_rec

def get_cassi_img(cube, mask, calibstack_row, calibstack_col, ksizes):
    '''
        Compute CASSI image from hyperspectral cube and mask.
        
        Inputs:
            cube: H x W x T hyperspectral cube. If shape is specified, create
                one from just repeated mask.
            mask: 2D mask of shape H x W
            calibstack_row: Calibration information for row correspondences
            calibstack_col: Calibration information for column correspondences
            ksizes: Blur sizes.
            
        Outputs:
            cassi_img: CASSI image
            
        NOTE: We will rely on cuda if available
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Convert to tensors
    if type(cube) is list:
        H, W, T = cube
        mask_ten = torch.tensor(mask, device=device)
        cube_ten = torch.zeros(T, 1, H, W, device=device)
        cube_ten[...] = mask_ten
    else:
        H, W, T = cube.shape
        cube *= mask[..., np.newaxis]
        cube_ten = torch.tensor(cube.transpose(2, 0, 1)[:, np.newaxis, :, :],
                                device=device)
    
    # Generate grid variable
    row_ten = torch.tensor(calibstack_row.transpose(2, 0, 1), device=device)
    row_ten = (2*row_ten - H)/H
    
    col_ten = torch.tensor(calibstack_col.transpose(2, 0, 1), device=device)
    col_ten = (2*col_ten - W)/W
    
    grid = torch.cat((col_ten[..., None], row_ten[..., None]), 3)
    
    # Create cassi images for each wavelength
    cassi_img_stk = torch.nn.functional.grid_sample(cube_ten, grid,
                                                    align_corners=False)
    # Blur the images
    for idx in range(T):
        win = torch.tensor(filtering.gaussdiskwin(ksizes[idx]).astype(np.float32),
                           device=device)
        win /= win.sum()
        h, _ = win.shape
        dh = h // 2
        cassi_img_stk[[idx], :, :, :] = torch.nn.functional.conv2d(cassi_img_stk[[idx], :, :, :],
                                                                   win[None, None, ...],
                                                                   padding=[dh, dh])
        
    # Squash
    cassi_img = cassi_img_stk.sum(0).squeeze().cpu()
    
    return cassi_img

def get_offset(cassi_img, streak_img):
    '''
        Compute offset from streak and cassi images
    '''
    Hc, Wc = streak_img.shape
    x, y = np.where(streak_img == 0)
    yf, xf = np.mgrid[:Hc, :Wc]
    
    offset_img = interpolate.griddata(np.vstack((y, x)).T,
                                      cassi_img[x, y],
                                      (xf.ravel(), yf.ravel()),
                                      method='nearest').reshape(Hc, Wc)
    
    return offset_img

def _get_cassi_cube(cassi_img, mask, Xl, Yl, vsize=3, ksizes=None):
    '''
        Convert cassi image to cassi cube.
        
        Inputs:
            cassi_img: Raw cassi image with offset removed (see get_offset)
            mask: Mask that was used to get CASSI image
            Xl, Yl: Calibration information to convert image to cube
            ksizes: convolution kernels to account for wavelength blur
            
        Outputs:
            cube: 3D cassi cube with spectrum at all points where mask == 1
    '''
    # We will do restricted interpolation
    xm, ym = np.where(mask > 0)
    
    Xlsub = Xl[xm, ym, :]
    Ylsub = Yl[xm, ym, :]
    
    Hl, Wl, T = Xl.shape
    cube = np.zeros((Hl, Wl, T), dtype=np.float32)
    
    # Filter the CASSI imagae
    cassi_img = signal.convolve2d(cassi_img, np.ones((vsize, 1)), 'same')
    
    if ksizes is None:
        vals = cv2.remap(cassi_img, Xlsub, Ylsub, cv2.INTER_LINEAR)
        cube[xm, ym, :] = vals
    else:
        for idx in range(T):
            win = filtering.gaussdiskwin(ksizes[idx])
            img = signal.convolve2d(cassi_img, win, mode='same')
            vals = cv2.remap(img, Xlsub[:, [idx]], Ylsub[:, [idx]], cv2.INTER_LINEAR)
            vals[np.isnan(vals)] = 0
            vals[np.isinf(vals)] = 0
            cube[xm, ym, idx] = vals.ravel()
        
    return cube

def get_cassi_cube(cassi_img, mask, Xl, Yl,  calibstack_row, calibstack_col,
                   vsize=3, ksizes=None):
    '''
        Convert cassi image to cassi cube.
        
        Inputs:
            cassi_img: Raw cassi image
            mask: Mask that was used to get CASSI image
            Xl, Yl: Calibration information to convert image to cube
            ksizes: convolution kernels to account for wavelength blur
            
        Outputs:
            cube: 3D cassi cube with spectrum at all points where mask == 1
    '''
    Hl, Wl, nwvl = Xl.shape
    
    # First compute streak image
    streak_img = get_cassi_img([Hl, Wl, nwvl],
                               mask,
                               calibstack_row,
                               calibstack_col,
                               ksizes)
    
    offset_img = get_offset(cassi_img, streak_img)
    
    cassi_img2 = cassi_img - offset_img
    cassi_img2[cassi_img2 < 0] = 0
    
    cube = _get_cassi_cube(cassi_img2, mask, Yl, Xl, vsize, None)
    
    return cube

def ssim(hsi1, hsi2, winsize=[11, 11], imgmode=False):
    '''
        Compute average SSIM for two hyperspectral images in a memory-efficient
        manner.

        Inputs:
            hsi1: First hyperspectral image
            hsi2: Second hyperspectral image
            winsize: Window size for computing mean and variances
            imgmode: If True, computes average SSIM at each pixel and returns
                a 2D image

        Outputs:
            ssimval: single value or 2D image with per-pixel SSIM
    '''

    H, W, T = hsi1.shape

    ssim_map = np.zeros((H, W), dtype=np.float32)

    C1 = 0.001**2
    C2 = 0.03**2

    for idx in range(T):
        mu1 = filtering.convsep(hsi1[:, :, idx], winsize)
        mu2 = filtering.convsep(hsi2[:, :, idx], winsize)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1*mu2

        var1 = filtering.convsep(hsi1[:, :, idx]*hsi1[:, :, idx], winsize)
        var2 = filtering.convsep(hsi2[:, :, idx]*hsi2[:, :, idx], winsize)
        var12 = filtering.convsep(hsi1[:, :, idx]*hsi2[:, :, idx], winsize)

        numer = (2*mu1_mu2  + C1)*(2*var12 + C2)
        denom = (mu1_sq + mu2_sq + C1)*(var1 + var2 + C2)

        ssim_map += numer/denom

    ssim_map /= T

    if not imgmode:
        ssim_map = ssim_map.mean()

    return ssim_map

if __name__ == '__main__':
    expname = '18.TriceraLED'
    root = 'D:/Dropbox/SASSI_Calibrate'
    rank = 6
    vsize = 3
    
    import sys
    sys.path.append('../')
    
    print('Loading calibration data')
    calibdata = io.loadmat('%s/calibration/calibstack.mat'%root)
    reverse_calibdata = io.loadmat('%s/calibration/reverse_lookup.mat'%root)
    
    print('Loading CASSI data')
    data = io.loadmat('%s/%s/adaptive_cassi.mat'%(root, expname))
    
    limits = data['limits'].ravel()
    mask = data['lcos_mask'].astype(np.float32)/255
    [Hl, Wl] = mask.shape
    nwvl = calibdata['wavelengths'].ravel().size
    cassi_img = data['cassi_img'].astype(np.float32)
    imrgb = data['guide_img_rec']
    wvl = calibdata['wavelengths'].ravel()
    
    Xl = reverse_calibdata['Xl'].astype(np.float32) - 1
    Yl = reverse_calibdata['Yl'].astype(np.float32) - 1
    
    calibstack_row = calibdata['calibstack_row'] - 1
    calibstack_col = calibdata['calibstack_col'] - 1
    
    ksizes = (np.linspace(1, 4, nwvl)**2)/4
    
    if False:
        print('Computing streak image')
        onecube = np.ones((Hl, Wl, nwvl), dtype=np.float32)
        streak_img = get_cassi_img(onecube,
                                mask,
                                calibdata['calibstack_row']-1,
                                calibdata['calibstack_col']-1,
                                ksizes)
        
        print('Computing offset image')
        offset_img = get_offset(cassi_img, streak_img)
        
        print('Creating HSI cube')
        cube = get_cassi_cube(cassi_img-offset_img, mask, Yl-1, Xl-1, vsize, None)
        
    cube = get_cassi_cube(cassi_img, mask, Xl, Yl, calibstack_row,
                          calibstack_col, vsize, ksizes)
    cube /= cube.max()
    h1, h2, w1, w2 = limits
    
    print('Updating superpixel membership')
    [L, N] = slic_update(imrgb, mask[h1:h2, w1:w2].astype(np.uint8), 10.0)
    
    print('Recovering cube')
    cube_rec = recon_superpixel(cube,
                                mask[h1:h2, w1:w2],
                                L,
                                imrgb.astype(np.float32).mean(2)/255,
                                N,
                                renorm=True,
                                net=None,
                                lr_filter=True,
                                rank=rank,
                                recon_3d=False)
    imrgb_rec = hyperspectral.hyper2rgb(cube_rec, wvl)
    plt.imshow(imrgb_rec)
    plt.show()