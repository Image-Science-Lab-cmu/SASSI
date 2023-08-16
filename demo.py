#!/usr/bin/env python

# System imports
import os
import sys
import time
import pickle
import importlib

# Scientific computing
import numpy as np
from scipy import io
from scipy import signal
from skimage.metrics import structural_similarity as ssim_func
from skimage import segmentation

# Plotting
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
plt.gray()

# PyTorch
import torch

# Our modules
from modules import pancnn
from modules import utils
from modules import hyperspectral
from modules import cassi
from modules import ssim

plt.rcParams["font.family"] = "Verdana"

if __name__ == '__main__':
	# Experiment constants
    expname = 'sample'          # Load this HSI for simulation
    scaling = 1                 # If code is slow, set scale to 0.5
    
    # Noise constants (all in photon)
    noise_snr = 5               # Readout noise in electrons
    tau = 100                   # Shot noise in electrons
    
    
    # Mask generation constants
    sp_frac = 0.1                   # Density of superpixels
    minsep = 31                     # Minimum horizontal separation between two openings
    
    # Neural network constants
    modelname = 'norecon_31_noise_100_40k_guided'   # Name of the pre-trained model
    modeltype = 'guided'                            # Type of NN architecture
    maskmode = True                                 # Used by NN architecture
    
    # Low rank filtering constants
    lr_filter = True                    # If true, perform low-rank projection
    rank = 6                            # Rank for low-rank projection
    
    # Data directories
    saveroot = 'results/%s'%expname     # Folder to save results
    dataroot = 'data'                   # Folder where data is located
    
    # Load data
    print('Loading data')
    data = io.loadmat('%s/%s/%s.mat'%(dataroot, expname, expname))
    info = io.loadmat('%s/%s/display_info.mat'%(dataroot, expname))

    hypercube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].ravel()

    hypercube = hypercube/hypercube.max()

    # Resize to prevent RAM overflow
    hypercube = cv2.resize(hypercube, None,
                           fx=(1.0/scaling),
                           fy=(1.0/scaling),
                           interpolation=cv2.INTER_AREA)
    H, W, T = hypercube.shape

    # Create a noisy version
    hsi_noisy = utils.measure(hypercube, noise_snr, tau).astype(np.float32)

    # Load networks
    print('Loading PanCNN models')
    net = pancnn.loadmodel(modelname, modeltype, maskmode).cuda()

    # Generate RGB image
    print('Generating sampling mask')
    imrgb = hyperspectral.hyper2rgb(hypercube, wavelengths, 1)

    imrgb_guide = pow(imrgb, 2.0)
    imrgb_guide = imrgb_guide/imrgb_guide.max()
    impan = imrgb_guide.mean(2)
    
    # Generate sampling mask, given super pixels
    centroid_mask, mask = cassi.get_sp_mask((imrgb*255).astype(np.uint8),
                                             minsep, frac=sp_frac,
                                             compactness=10)
    
    print('Updating superpixel membership')
    L2, N2 = cassi.slic_update((255*imrgb).astype(np.uint8),
                               centroid_mask, compactness=20)

    print('Superpixel reconstruction')
    hsi_recon1 = cassi.recon_superpixel(hsi_noisy*mask[:, :, np.newaxis],
                                        mask.astype(np.float32),
                                        L2,
                                        impan.astype(np.float32),
                                        N2,
                                        lr_filter=lr_filter,
                                        rank=rank,
                                        renorm=False)
    imrgb_rec1 = hyperspectral.hyper2rgb(hsi_recon1, wavelengths, 1)

    print('Guided reconstruction')
    hsi_recon2 = cassi.recon_guided(hsi_noisy*mask[:, :, np.newaxis],
                                    impan, mask,
                                    lr_filter=lr_filter,
                                    rank=rank,
                                    winsize=tuple(net.conv1.weight.shape[2:]))
    hsi_recon2[np.isnan(hsi_recon2)] = 0
    hsi_recon2[np.isinf(hsi_recon2)] = 0

    imrgb_rec2 = hyperspectral.hyper2rgb(hsi_recon2, wavelengths, 1)

    print('Superpixelation + NN filtering')
    hsi_recon3 = cassi.recon_superpixel(hsi_noisy*mask[:, :, np.newaxis],
                                        mask.astype(np.float32),
                                        L2,
                                        impan.astype(np.float32),
                                        N2,
                                        net=net,
                                        recon_3d=(modeltype=='guided3d'),
                                        lr_filter=lr_filter,
                                        rank=rank,
                                        renorm=False)
    # Clip results to 0, 1
    hsi_recon3 = np.clip(hsi_recon3, 0, 1)
    
    cube_erg = np.sqrt((hypercube**2).sum(2))
    cube_erg += 1e-2*hypercube.max()
    err1 = np.sqrt(((hypercube - hsi_recon1)**2).mean(2))
    err2 = np.sqrt(((hypercube - hsi_recon2)**2).mean(2))
    err3 = np.sqrt(((hypercube - hsi_recon3)**2).mean(2))

    imrgb_rec3 = hyperspectral.hyper2rgb(hsi_recon3, wavelengths, 1)

    fig = plt.gcf()

    print('Computing metrics')
    rsnr1 = utils.asnr(hypercube, hsi_recon1, True)
    rsnr2 = utils.asnr(hypercube, hsi_recon2, True)
    rsnr3 = utils.asnr(hypercube, hsi_recon3, True)

    ten = torch.tensor

    ssim1 = ssim_func(hypercube, hsi_recon1)
    ssim2 = ssim_func(hypercube, hsi_recon2)
    ssim3 = ssim_func(hypercube, hsi_recon3)
    
    w1, h1, dw, dh = (info['limits'].ravel()/scaling).astype(int)
    w2 = w1 + dw
    h2 = h1 + dh
    
    # Write information
    os.makedirs(saveroot, exist_ok=True)
    
    plt.imsave('%s/im_sp.png'%saveroot, imrgb_rec1)
    plt.imsave('%s/im_gu.png'%saveroot, imrgb_rec2)
    plt.imsave('%s/im_nn.png'%saveroot, imrgb_rec3)
    
    plt.imsave('%s/err_sp.png'%saveroot, err1/cube_erg, vmin=0, vmax=1, cmap='jet')
    plt.imsave('%s/err_gu.png'%saveroot, err2/cube_erg, vmin=0, vmax=1, cmap='jet')
    plt.imsave('%s/err_nn.png'%saveroot, err3/cube_erg, vmin=0, vmax=1, cmap='jet')
    
    im_boundary = segmentation.mark_boundaries(imrgb, L2, color=(0, 0, 0))
    plt.imsave('%s/im_boundary.png'%saveroot, np.clip(im_boundary, 0, 1))
    plt.imsave('%s/gt.png'%saveroot, imrgb)
    plt.imsave('%s/mask.png'%saveroot,
               signal.convolve2d(mask, np.ones((4, 4)), mode='same'),
               vmin=0, vmax=1)

    plt.subplot(2, 4, 1)
    plt.imshow(imrgb)
    plt.xticks([], []); plt.yticks([], [])
    plt.title('Ground truth RGB')
    
    plt.subplot(2, 4, 2)
    plt.imshow(hsi_recon1[..., T//2], cmap='gray')
    plt.xticks([], []); plt.yticks([], [])
    plt.title('Superpixel (%.2f dB; %.2f)'%(rsnr1, ssim1))

    plt.subplot(2, 4, 3)
    plt.imshow(hsi_recon2[..., T//2], cmap='gray')
    plt.xticks([], []); plt.yticks([], [])
    plt.title('Guided (%.2f dB; %.2f)'%(rsnr2, ssim2))

    plt.subplot(2, 4, 4)
    plt.imshow(hsi_recon3[..., T//2], cmap='gray')
    plt.xticks([], []); plt.yticks([], [])
    plt.title('Superpixel+NN (%.2f dB; %.2f)'%(rsnr3, ssim3))
    
    plt.subplot(2, 4, 5)
    plt.imshow(hypercube[..., T//2]); plt.title('GT')

    plt.subplot(2, 2, 4)
    lambda_w, lambda_h = ((info['points'][0, :]-1)/scaling).astype(int)
    plt.plot(hypercube[lambda_h, lambda_w, :], 'x-', label='Ground truth')
    plt.plot(hsi_recon1[lambda_h, lambda_w, :], '^--', label='Superpixels')
    plt.plot(hsi_recon2[lambda_h, lambda_w, :], 'p-.', label='Naive guided filter')
    plt.plot(hsi_recon3[lambda_h, lambda_w, :], '+-', label='Learnt guided filter')

    plt.grid(True)
    plt.legend()

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    fig.savefig('%s/sim_%dx.png'%(saveroot, scaling), dpi=100)
    
    fig2 = plt.figure(figsize=(9, 4))
    plt.plot(wavelengths, hypercube[lambda_h, lambda_w, :], 'x-', label='Ground truth')
    plt.plot(wavelengths, hsi_recon1[lambda_h, lambda_w, :], '^--', label='Superpixels')
    plt.plot(wavelengths, hsi_recon2[lambda_h, lambda_w, :], 'p-.', label='Naive guided filter')
    plt.plot(wavelengths, hsi_recon3[lambda_h, lambda_w, :], '+-', label='Learnt guided filter')
    plt.xlabel('$\lambda$ (nm)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, linewidth=0.5, color=(0.9, 0.9, 0.9))
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('%s/spectrum.svg'%saveroot)
    
    # Save data for simulation comparisons
    mdict = {'imrgb_nn': imrgb_rec3,
             'spec_nn': hsi_recon3[lambda_h, lambda_w, :],
             'psnr_nn': rsnr3,
             'ssim_nn': ssim3}
    io.savemat('%s/visual_nn.mat'%saveroot, mdict)
    
