'''
motion_field_estimation_utils.py

Utilities for estimating the motion field between two 2D (single channel) same size images \
    by using the gradient constraint equation
'''

import numpy as np
from scipy.ndimage import shift, rotate
from skimage.transform import warp
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def ImageTranslate(image_in, shift_row, shift_col): 
    image_out = shift(image_in, (shift_row, shift_col));
    return image_out;

def ImageRotate(image_in, rot_angle):   
    image_out = rotate(image_in, rot_angle, reshape=False);
    return image_out;
    
def MotionCorrection(image_in, delta_y, delta_x):
    Nrows, Ncols = image_in.shape;
    row_coords, col_coords = np.meshgrid(np.arange(Nrows), np.arange(Ncols), indexing='ij');
    image_out = warp(image_in, np.array([row_coords + delta_y, col_coords + delta_x]), mode='edge');
    return image_out;
    
def _MotionFieldEstimationSingleStep(image0, image1, neighborhood_size = 7, sigma = 3, reg_coef = 0):
 
    delta_y = np.zeros(image0.shape);
    delta_x = np.zeros(image0.shape);
    
    # derivative kernels
    kernel_x = np.array([[-1, 0, 1]]) / 2;
    kernel_y = kernel_x.T;
    
    # smothing to reduce the higher order terms in the Taylor expansion 
    image0_f = gaussian_filter(image0, sigma, mode = 'constant', cval = 0);
    image1_f = gaussian_filter(image1, sigma, mode = 'constant', cval = 0);
    
    # spatial and temporal image gradients
    Ix = convolve2d(image0_f, kernel_x, boundary = 'fill', fillvalue = 0, mode = 'same');
    Iy = convolve2d(image0_f, kernel_y, boundary = 'fill', fillvalue = 0, mode = 'same');
    It = image1_f - image0_f;
    
    IxIx = Ix * Ix;
    IxIy = Ix * Iy;
    IyIy = Iy * Iy;
    IxIt = Ix * It;
    IyIt = Iy * It;
    
    # smothing at this stage is equivalent to giving a higher weighting to the center 
    # of the neighborhood
    IxIx_f = gaussian_filter(IxIx, sigma, mode = 'constant', cval = 0);
    IxIy_f = gaussian_filter(IxIy, sigma, mode = 'constant', cval = 0);
    IyIy_f = gaussian_filter(IyIy, sigma, mode = 'constant', cval = 0);
    IxIt_f = gaussian_filter(IxIt, sigma, mode = 'constant', cval = 0);
    IyIt_f = gaussian_filter(IyIt, sigma, mode = 'constant', cval = 0);
    
    nb = int((neighborhood_size - 1) / 2);

    for ii in range(nb, image0.shape[0] - nb):
        for jj in range(nb, image0.shape[1] - nb):
            
            # elements of matrix A (2 x 2)
            a = IxIx_f[ii - nb:ii + nb + 1, jj - nb:jj + nb + 1];
            a = np.sum(a, axis = (0, 1));    
            b = IxIy_f[ii - nb:ii + nb + 1, jj - nb:jj + nb + 1];
            b = np.sum(b, axis = (0, 1));   
            c = b;    
            d = IyIy_f[ii - nb:ii + nb + 1, jj - nb:jj + nb + 1];
            d = np.sum(d, axis = (0, 1));             
            
            # elements of vector B (2, 1)
            f = IxIt_f[ii - nb:ii + nb + 1, jj - nb:jj + nb + 1];
            f = np.sum(f, axis = (0, 1));    
            g = IyIt_f[ii - nb:ii + nb + 1, jj - nb:jj + nb + 1];
            g = np.sum(g, axis = (0, 1)); 
            
            # system of linear eqs
            A = np.array([[a, b], [c, d]]);
            B = -np.array([[f], [g]]);
            
            # normal equation with Tikhonov regularization
            X = np.linalg.solve(A.T @ A + reg_coef * np.eye(2,2), A.T @ B);            
            delta_x[ii, jj], delta_y[ii, jj] = -X;
            
    return delta_y, delta_x;

def MotionFieldEstimation(image0, image1, neighborhood_size = 7, sigma = 3, reg_coef = 0, Niter = 1):
    
    delta_y_iter = np.zeros(image0.shape);
    delta_x_iter = np.zeros(image0.shape);
    
    for i in range(Niter):
        
        if i >= 1:
            neighborhood_size = 5;
            sigma = 3;
        
        delta_y, delta_x = _MotionFieldEstimationSingleStep(image0, image1, neighborhood_size, sigma, reg_coef);
        image1 = MotionCorrection(image1, delta_y, delta_x);
        delta_y_iter += delta_y;
        delta_x_iter += delta_x;     
        
    return delta_y_iter, delta_x_iter;
            
            
            
            
            
            
            
            