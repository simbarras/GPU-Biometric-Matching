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

model_path = 'dataset_i/maximum_curvature_old/5_left_ring_1_cam1'
probe_path = 'dataset_i/maximum_curvature_old/5_left_ring_2_cam1'

model_img_path = 'dataset_i/5_left_ring_1_cam1.png'
probe_img_path = 'dataset_i/5_left_ring_2_cam1.png'

model_img = Image.open(model_img_path)


probe_img = Image.open('dataset_i/3_right_index_2_cam1.png')

W_original = np.asarray(probe_img)
# W, mask = regiongrow(W, roi=(40, 210, 10, 360))
# W = remove_static_mask(W, 1)
# W, mask = fingerfocus(W, roi=(40, 210, 40, 360))
#W_, mask = extract_features(W, mask, "none")

plt.imshow(W_original)
plt.show()

def segmentation(W):
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
    W[W > 180] = 0
    W[W > 0] = 1

    # W_mask = W[W > 230]
    # W[W_mask] = 0

    image_show(W)
    plt.show()

    print(np.min(gradient), np.max(gradient))
    grad_thresh = 7
    W[gradient > grad_thresh] = 0

    image_show(gradient)
    plt.show()


    return W


# get rough mask
W = segmentation(W_original.copy())

W = ndimage.binary_erosion(W, structure=[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]], iterations=5).astype(W.dtype)
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

W = ndimage.binary_dilation(W, structure=[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]], iterations=5).astype(W.dtype)
image_show(W)
plt.show()

# W = ndimage.binary_dilation(W, iterations=5).astype(W.dtype)
# image_show(W)
# plt.show()


W = convex_hull_image(W)
image_show(W)
plt.show()

W = remove_static_mask(W, 1)
image_show(W)
plt.show()

plt.imshow(W_original)
plt.imshow(W, alpha=0.4)
plt.show()