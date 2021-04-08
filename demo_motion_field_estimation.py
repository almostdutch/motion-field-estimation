'''
demo_motion_field_estimation.py

This demo shows how to estimate motion field between two 2D (single channel) same size images\
    by using the gradient constraint equation with iterative refinement
    
The implemented motion model is based on the assumption that all pixels in the \
    local neighborhood undergo the same translational motion, and hence have the same motion vectors
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
import random
from motion_field_estimation_utils import ImageTranslate, ImageRotate, \
    MotionCorrection, MotionFieldEstimation

# range for taking a random shift (pixels) and rotation angle (degrees)
limit_left = -5.00;
limit_right = 5.00;

mu, sigma = 0, 4; # mean and standard deviation of added Gaussian noise
Niter = 50; # number of iterations

# load test image
image_ref = np.array(mpimg.imread('test_image.gif'));
image_ref = image_ref.astype(np.float64);
Nrows, Ncols = image_ref.shape;

# generate random independent row and column pixel shift, and rotation angle
shift_row_rand = round(random.uniform(limit_left, limit_right), 2);
shift_col_rand = round(random.uniform(limit_left, limit_right), 2);
rot_angle_rand = round(random.uniform(limit_left, limit_right), 2);

print('Given rotation angle: ' + str(rot_angle_rand));
print('Given translation Row: ' + str(shift_row_rand) + ' Col: ' + str(shift_col_rand));

# generated dummy image, shifted and rotated
image_shifted = ImageTranslate(image_ref.copy(), shift_row_rand, shift_col_rand);
image_rotated = ImageRotate(image_shifted.copy(), rot_angle_rand);

# add independent Gaussian noise for reference image (image_ref) and rotated image (image_rotated)
image_ref += np.random.normal(mu, sigma, size = (Nrows, Ncols));
image_rotated += np.random.normal(mu, sigma, size = (Nrows, Ncols));

image0 = image_ref / 255;
image1 = image_rotated / 255;

neighborhood_size = 21;
sigma = 5;
reg_coef = 10e-3;
Niter = 1;
delta_y, delta_x = MotionFieldEstimation(image0, image1, neighborhood_size, sigma, reg_coef, Niter);

# show original data
fig_width, fig_height = 10, 5;
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image0, cmap='gray')
ax1.set_title("image0")
ax1.set_axis_off()

ax2.imshow(image1, cmap='gray')
ax2.set_title("image1")
ax2.set_axis_off()

ax3.imshow(image0 - image1)
ax3.set_title("difference")
ax3.set_axis_off()

image_sequence = np.zeros((Nrows, Ncols, 3));
image_sequence[:, :, 0] = image1; # red channel
image_sequence[:, :, 1] = image0; # green channel
image_sequence[:, :, 2] = image0; # blue channel

ax4.imshow(image_sequence)
ax4.set_title("RGB sequence")
ax4.set_axis_off()
plt.tight_layout()

# magnitude of motion field
magn_motion_field = np.sqrt(delta_x ** 2 + delta_y ** 2);

# show the estimated motion field
fig, (ax1) = plt.subplots(1, 1, figsize=(fig_width, fig_height));

Nvectors = 50;  # number of vectors to be displayed along each image dimension
step = max(Nrows//Nvectors, Ncols//Nvectors);

y, x = np.mgrid[:Nrows:step, :Ncols:step];
delta_x_ = delta_x[::step, ::step];
delta_y_ = delta_y[::step, ::step];

ax1.imshow(magn_motion_field)
ax1.quiver(x, y, delta_x_, delta_y_, color='r', units='dots', angles='xy', scale_units='xy')
ax1.set_title("Motion field: magnitude and vector field")
ax1.set_axis_off()
fig.tight_layout()

# show motion corrected data
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image0, cmap='gray')
ax1.set_title("image0")
ax1.set_axis_off()

image1 = MotionCorrection(image1, delta_y, delta_x);
ax2.imshow(image1, cmap='gray')
ax2.set_title("image1 motion corrected")
ax2.set_axis_off()

ax3.imshow(image0 - image1)
ax3.set_title("difference")
ax3.set_axis_off()

image_sequence = np.zeros((Nrows, Ncols, 3));
image_sequence[:, :, 0] = image1; # red channel
image_sequence[:, :, 1] = image0; # green channel
image_sequence[:, :, 2] = image0; # blue channel

ax4.imshow(image_sequence)
ax4.set_title("RGB sequence")
ax4.set_axis_off()
plt.tight_layout()