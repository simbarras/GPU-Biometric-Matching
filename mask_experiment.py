from scipy import ndimage

from alignment_evaluation import *
from resources import remove_static_mask
import numpy as np
import cv2 as cv
import matplotlib.pyplot as pltimport, skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
import scipy.ndimage as si


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

model_path = 'dataset_ii/maximum_curvature_old/5_left_ring_1_cam1'
probe_path = 'dataset_ii/maximum_curvature_old/5_left_ring_2_cam1'

model_img_path = 'dataset_ii/18_left_index_3_cam1.png'
probe_img_path = 'dataset_ii/18_left_index_4_cam1.png'

model_img = Image.open(model_img_path)

# [
#     [a, b, c]
#     [d, e, f]
#     [0, 0, 1]
# ]
#
# -> a, b, d, e parameterized with one variable
# a = cos alpha
# b = -sin alpha
# [x, y, 1]



probe_img = Image.open('dataset_ii/10_right_index_1_cam2.png')

W_original = np.asarray(probe_img)
# W, mask = regiongrow(W, roi=(40, 210, 10, 360))
# W = remove_static_mask(W, 1)
# W, mask = fingerfocus(W, roi=(40, 210, 40, 360))
#W_, mask = extract_features(W, mask, "none")

plt.imshow(W_original)
plt.show()

bright = np.zeros_like(W_original)

def segmentation(W, bright):
    ######## uncomment to show histogram
    # fig, ax = plt.subplots(1, 1)
    # ax.hist(W.ravel(), bins=32, range=[0, 256])
    # ax.set_xlim(0, 256)
    # plt.show()
    gx, gy = np.gradient(W)
    gradient = np.hypot(gx, gy)

    img_blur = cv.blur(W, (200,5))
    width = img_blur.shape[1]
    height = img_blur.shape[0]
    center_x = round(width / 2)
    center_y = round(height / 2)
    image_show(img_blur)
    plt.show()
    thresh = img_blur[center_y, center_x]
    thresh = min(thresh - 10, 50)
    print(thresh)
    W[W < thresh] = 0
    bright[W > 230] = 1
    W[W > 180] = 0

    image_show(bright)
    plt.show()


    W[W > 0] = 1

    # W_mask = W[W > 230]
    # W[W_mask] = 0

    image_show(W)
    plt.show()

    print(np.min(gradient), np.max(gradient))
    grad_thresh = 7
    W[gradient > grad_thresh] = 0

    image_show(W)
    plt.show()


    return W, bright


# get rough mask
W, bright = segmentation(W_original.copy(), bright)

# erosions:
diag_right = [
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1]
]

W = ndimage.binary_erosion(W, structure=diag_right, iterations=2).astype(W.dtype)
image_show(W)
plt.show()

# use all connected components with center of image
width = W.shape[1]
height = W.shape[0]
start_x = round(width / 2)
start_y = round(height / 2)
blobs, labels = si.label(W, structure=np.array([[0, 1, 0],
                                               [1, 1, 1],
                                               [0, 1, 0]]))
W[blobs != blobs[start_y, start_x]] = 0
image_show(W)
plt.show()

W = ndimage.binary_dilation(W, structure=diag_right, iterations=2).astype(W.dtype)
image_show(W)
plt.show()

# W = ndimage.binary_dilation(W, iterations=5).astype(W.dtype)
# image_show(W)
# plt.show()


W = convex_hull_image(W)
image_show(W)
plt.show()

# W = remove_static_mask(W, 1)
# image_show(W)
# plt.show()

image_show(bright)
plt.show()
bright = ndimage.binary_dilation(bright, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], iterations=5).astype(bright.dtype)
W[bright == 1] = 0

plt.imshow(W_original)
plt.imshow(W, alpha=0.4)
plt.show()

W, mask = extract_features(W_original, W)
plt.imshow(W_original)
plt.imshow(W, alpha=0.2)
plt.imshow(mask, alpha=0.4)
plt.show()