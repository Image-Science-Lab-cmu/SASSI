# cython: language_level=3

'''
    Cythonized functions for handling adaptive CASSI sub-routines
'''

import numpy as np
import cv2
from cython.parallel import parallel, prange

# Compile time optimizations
cimport numpy as np
cimport cython

# We will mostly use UINT8
DTYPE_UINT8 = np.uint8
DTYPE_UINT16 = np.uint16
DTYPE_FLOAT32 = np.float32
DTYPE_INT16 = np.int16

ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.uint16_t DTYPE_UINT16_t
ctypedef np.float32_t DTYPE_FLOAT32_t
ctypedef np.int16_t DTYPE_INT16_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sanitize(np.ndarray[DTYPE_UINT8_t, ndim=2] mask, int minsep):
    '''
        Function to sanitize a CASSI mask such that no two openings along a 
        row are separated by less than minsep

        Inputs:
            mask: Binary mask with 1 at spatial location having openings
            minsep: Minimum separation between two openings along the row

        Outputs:
            mask: Sanitized CASSI mask
    '''
    # Declare all variables here
    cdef int H
    cdef int W
    cdef int h
    cdef int next_idx
    cdef int next_idx2
    cdef int iidx

    H = mask.shape[0]
    W = mask.shape[1]

    for h in range(H):
        next_idx = 0

        while next_idx < W-1:
            if mask[h, next_idx] == 1:
                next_idx2 = min(W, next_idx + minsep)
                if h < H-1:
                    for iidx in range(next_idx+1, next_idx2):
                        mask[h+1, iidx] = mask[h, iidx]
				
                for iidx in range(next_idx+1, next_idx2):
                    mask[h, iidx] = 0
                next_idx += minsep
                
            else:
                next_idx += 1

    return mask

@cython.boundscheck(False)
@cython.wraparound(False)
def intensify(np.ndarray[DTYPE_UINT8_t, ndim=2] mask,
              np.ndarray[DTYPE_UINT8_t, ndim=2] mask_conv,
              int minsep):
    '''
        Function to increase density of openings while ensuring that openings
        are separated by minsep.

        Inputs:
            mask: Binary mask
            mask_conv: Dilated mask for fast computing
            minsep: Minimum separation

    '''
    cdef int H
    cdef int W
    cdef int h
    cdef int next_idx

    H = mask.shape[0]
    W = mask.shape[1]

    for h in range(H):
        next_idx = 0

        while next_idx < W:
            if mask[h, next_idx] == 1:
                next_idx +=  minsep
            else:
                if mask_conv[h, next_idx] == 0:
                    mask[h, next_idx] = 1
                    next_idx += minsep
                else:
                    next_idx += 1

    return mask

@cython.boundscheck(False)
@cython.wraparound(False)
def random_intensify(np.ndarray[DTYPE_UINT8_t, ndim=2] mask,
                     int minsep):
    '''
        Function to increase density of openings while ensuring that openings
        are separated by minsep via randomly increasing sampling.

        Inputs:
            mask: Binary mask
            minsep: Minimum separation

    '''
    cdef int H
    cdef int W
    cdef int h
    cdef int w

    cdef int w1
    cdef int w2
    cdef int idx

    H = mask.shape[0]
    W = mask.shape[1]

    # Create a random mask and assign memmory view
    random_mask = (np.random.rand(H, W) < 2/minsep).astype(np.uint8)
    cdef DTYPE_UINT8_t[:, :] random_mask_view = random_mask

    mask_intense = np.zeros((H, W), dtype=DTYPE_UINT8)
    cdef DTYPE_UINT8_t[:, :] mask_intense_view = mask_intense

    # Sanitize the random mask
    sanitize(random_mask, minsep)

    # Step 1 -- OR the two masks
    for h in prange(H, nogil=True):
        for w in range(W):
            mask_intense_view[h, w] = random_mask_view[h, w] | mask[h, w]

    # Step 2 -- isolate points at mask sampling locations
    for h in prange(H, nogil=True):
        for w in range(W):
            if mask[h, w] == 1:
                w1 = max(w - minsep + 1, 0)
                w2 = min(w + minsep, W)
                for idx in range(w1, w2):
                    mask_intense_view[h, idx] = 0
                mask_intense_view[h, w] = 1

    return mask_intense

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_dist_cp(np.ndarray[DTYPE_FLOAT32_t, ndim=1] centroid_xy,
                 np.ndarray[DTYPE_FLOAT32_t, ndim=3] imlabxy_patch,
                 float compactness, int S):
    '''
        Function to rapidly compute distance from a centroid over a patch
    '''
    # Declare all variables ahead
    cdef int H
    cdef int W
    cdef int h
    cdef int w
    cdef float C

    H = imlabxy_patch.shape[0]
    W = imlabxy_patch.shape[1]
    C = compactness/S

    # Create new matrix to store distances
    dist = np.zeros((H, W), dtype=DTYPE_FLOAT32)

    # Creating a data view will make all operations much faster
    cdef DTYPE_FLOAT32_t[:, :] dist_view = dist

    # Now run through all variables
    for h in prange(H, nogil=True):
        for w in range(W):
            # Unroll the whole computation
            dist_view[h, w] = ((imlabxy_patch[h, w, 0] - centroid_xy[0])**2 + \
                               (imlabxy_patch[h, w, 1] - centroid_xy[1])**2 + \
                               (imlabxy_patch[h, w, 2] - centroid_xy[2])**2 + \
                             C*(imlabxy_patch[h, w, 3] - centroid_xy[3])**2 + \
                             C*(imlabxy_patch[h, w, 4] - centroid_xy[4])**2)

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def recon_superpixel(np.ndarray[DTYPE_FLOAT32_t, ndim=3] hsi_inflated,
                     np.ndarray[DTYPE_FLOAT32_t, ndim=2] mask,
                     np.ndarray[DTYPE_UINT16_t, ndim=2] labels,
                     np.ndarray[DTYPE_FLOAT32_t, ndim=2] impan, int N):
    '''
        Reconstruct the hyperspectral image using superpixel propagation
        approach

        Inputs:
            hsi_inflated: HxWxT cube with spectrum only at sampled locations
            mask: HxW sampling mask
            labels: HxW image with membership information
            impan: Grayscale guide/panchromatic image
            N: Number of superpixels
            renormalize: If true, ensure the final grayscale image matches
                the panchromatic image

        Outputs:
            hsi_rec: Reconstructed hsi
    '''

    # Declare all variables ahead
    cdef int H
    cdef int W
    cdef int T
    cdef int h
    cdef int w
    cdef int t
    cdef int c
    cdef int lbl
    cdef int maxsup
    cdef float denom

    H = hsi_inflated.shape[0]
    W = hsi_inflated.shape[1]
    T = hsi_inflated.shape[2]

    maxsup = (H*W*10)//N

     # Create new matrix to store reconstruction
    hsi_rec = np.zeros((H, W, T), dtype=DTYPE_FLOAT32)

    # Creating a data view will make all operations much faster
    cdef DTYPE_FLOAT32_t[:, :, :] hsi_rec_view = hsi_rec

    # Temporary storage variables
    cnt_array = np.zeros(N, dtype=DTYPE_UINT16)
    mmb_h_array = np.zeros((N, maxsup), dtype=DTYPE_UINT16)
    mmb_w_array = np.zeros((N, maxsup), dtype=DTYPE_UINT16)
    spec_avg_array = np.zeros((N, T), dtype=DTYPE_FLOAT32)

    cdef DTYPE_UINT16_t[:] cnt_array_view = cnt_array
    cdef DTYPE_UINT16_t[:, :] mmb_h_array_view = mmb_h_array
    cdef DTYPE_UINT16_t[:, :] mmb_w_array_view = mmb_w_array
    cdef DTYPE_FLOAT32_t[:, :] spec_avg_array_view = spec_avg_array

    # First create membership variables
    for h in prange(H, nogil=True):
        for w in range(W):
            lbl = labels[h, w]
            mmb_h_array_view[lbl, cnt_array_view[lbl]] = h
            mmb_w_array_view[lbl, cnt_array_view[lbl]] = w
            cnt_array_view[lbl] += 1

    # Next compute average spectrum for each superpixel
    for lbl in prange(N, nogil=True):
        denom = 1e-5
        for c in range(cnt_array_view[lbl]):
            h = mmb_h_array_view[lbl, c]
            w = mmb_w_array_view[lbl, c]

            denom += impan[h, w]*mask[h, w]

            for t in range(T):
                spec_avg_array_view[lbl, t] += hsi_inflated[h, w, t]*mask[h, w]

        for t in range(T):
            spec_avg_array_view[lbl, t] /= denom

    # Now reconstruct per each super pixel
    for h in prange(H, nogil=True):
        for w in range(W):
            lbl = labels[h, w]
            for t in range(T):
                hsi_rec_view[h, w, t] = impan[h, w]*spec_avg_array_view[lbl, t]

    return hsi_rec

@cython.boundscheck(False)
@cython.wraparound(False)
def rgb2lab(np.ndarray[DTYPE_UINT8_t, ndim=3] imrgb):
    '''
        Convert RGB image to LAB image
    '''
    cdef int H
    cdef int W
    cdef int h
    cdef int w

    H = imrgb.shape[0]
    W = imrgb.shape[1]

    imlab = np.zeros((H, W, 3), dtype=DTYPE_FLOAT32)

    cdef DTYPE_FLOAT32_t[:, :, :] imlab_view = imlab

    for h in prange(H, nogil=True):
        for w in range(W):
            imlab_view[h, w, 0] = 0.412453*imrgb[h, w, 0] + \
                                  0.357580*imrgb[h, w, 1] + \
                                  0.180423*imrgb[h, w, 2]

            imlab_view[h, w, 1] = 0.212671*imrgb[h, w, 0] + \
                                  0.715160*imrgb[h, w, 1] + \
                                  0.072169*imrgb[h, w, 2]

            imlab_view[h, w, 2] = 0.019334*imrgb[h, w, 0] + \
                                  0.119193*imrgb[h, w, 1] + \
                                  0.950227*imrgb[h, w, 2]

    return imlab

@cython.boundscheck(False)
@cython.wraparound(False)
def slic_update(np.ndarray[DTYPE_UINT8_t, ndim=3] imrgb,
                np.ndarray[DTYPE_UINT8_t, ndim=2] mask,
                float compactness):
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
    cdef int H
    cdef int W
    cdef int h
    cdef int w
    cdef int N
    cdef int S
    cdef int idx
    cdef float dist_val
    cdef float C

    cdef int hmin
    cdef int hmax
    cdef int wmin
    cdef int wmax

    H = mask.shape[0]
    W = mask.shape[1]

    N = mask.sum()
    S = int(np.sqrt(H*W/N))
    C = compactness/S


    imlab = np.zeros((H, W, 3), dtype=np.float32)

    ch = np.zeros(N, dtype=np.uint16)
    cw = np.zeros(N, dtype=np.uint16)
    centroids_labxy = np.zeros((N, 5), dtype=np.float32)
    dist_matrix = np.ones((H, W), dtype=np.float32)*float('inf')
    L = np.ones((H, W), dtype=np.uint16)

    # Extract memory views
    cdef DTYPE_FLOAT32_t[:, :, :] imlab_view = imlab
    cdef DTYPE_FLOAT32_t[:, :] centroids_view = centroids_labxy
    cdef DTYPE_FLOAT32_t[:, :] dist_matrix_view = dist_matrix
    cdef DTYPE_UINT16_t[:, :] L_view = L
    cdef DTYPE_UINT16_t[:] ch_view = ch
    cdef DTYPE_UINT16_t[:] cw_view = cw

    for h in prange(H, nogil=True):
        for w in range(W):
            imlab_view[h, w, 0] = 0.412453*imrgb[h, w, 0] + \
                                  0.357580*imrgb[h, w, 1] + \
                                  0.180423*imrgb[h, w, 2]

            imlab_view[h, w, 1] = 0.212671*imrgb[h, w, 0] + \
                                  0.715160*imrgb[h, w, 1] + \
                                  0.072169*imrgb[h, w, 2]

            imlab_view[h, w, 2] = 0.019334*imrgb[h, w, 0] + \
                                  0.119193*imrgb[h, w, 1] + \
                                  0.950227*imrgb[h, w, 2]

    idx = 0
    for h in range(H):
        for w in range(W):
            if mask[h, w] == 1:
                ch_view[idx] = h
                cw_view[idx] = w
                idx += 1

    # Now run over patches
    for idx in range(N):
        hmin = max(0, ch_view[idx] - 2*S); hmax = min(H, ch_view[idx] + 2*S)
        wmin = max(0, cw_view[idx] - 2*S); wmax = min(W, cw_view[idx] + 2*S)

        centroids_view[idx, 0] = imlab_view[ch_view[idx], cw_view[idx], 0]
        centroids_view[idx, 1] = imlab_view[ch_view[idx], cw_view[idx], 1]
        centroids_view[idx, 2] = imlab_view[ch_view[idx], cw_view[idx], 2]
        centroids_view[idx, 3] = ch_view[idx]
        centroids_view[idx, 4] = cw_view[idx]

        for h in prange(hmin, hmax, nogil=True):
            for w in range(wmin, wmax):
                # Unroll the distance computing here
                dist_val = ((imlab_view[h, w, 0] - centroids_view[idx, 0])**2 + \
                            (imlab_view[h, w, 1] - centroids_view[idx, 1])**2 + \
                            (imlab_view[h, w, 2] - centroids_view[idx, 2])**2 + \
                            C*(h - centroids_view[idx, 3])**2 + \
                            C*(w - centroids_view[idx, 4])**2)

                if dist_val < dist_matrix_view[h, w]:
                    L_view[h, w] = idx
                    dist_matrix_view[h, w] = dist_val

    return L, N
