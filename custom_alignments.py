import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
import matplotlib as mpl

import alignment_evaluation
from alignment_evaluation import *
# importing the module
import cv2
mpl.use('WebAgg')

# load images
# model_path = 'dataset_i/maximum_curvature/3_left_middle_1_cam1'
# probe_path = 'dataset_i/maximum_curvature/3_left_middle_2_cam1'
# model_img_path = 'dataset_i/3_left_middle_1_cam1.png'
# probe_img_path = 'dataset_i/3_left_middle_2_cam1.png'

model_path = 'dataset_i/maximum_curvature/5_left_ring_1_cam1'
probe_path = 'dataset_i/maximum_curvature/5_left_ring_2_cam1'
model_img_path = 'dataset_i/5_left_ring_1_cam1.png'
probe_img_path = 'dataset_i/5_left_ring_2_cam1.png'



model = np.load(model_path + ".npy")
model_mask = np.load(model_path + "_mask.npy")
probe = np.load(probe_path + ".npy")
probe_mask = np.load(probe_path + "_mask.npy")

model_img = Image.open(model_img_path)
probe_img = Image.open(probe_img_path)


def remove_mask(img, mask):
    shape = img.shape
    filtered = np.zeros(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x] == 1:
                filtered[y, x] = img[y, x]
    return filtered

def visualize_diff(a, b):
    return 2 * a + b

def visualize_xor(a, b):
    return np.bitwise_xor(a.astype(int), b.astype(int))

def plot_images_overlay():
    fig = plt.figure(0, figsize=(10, 7))
    plt.imshow(model_img)
    plt.imshow(model, cmap="gray", alpha=0.25)

    fig = plt.figure(1, figsize=(10, 7))
    plt.imshow(probe_img)
    plt.imshow(probe, cmap="gray", alpha=0.25)

    fig = plt.figure(2, figsize=(10, 7))
    plt.imshow(model_img)
    plt.imshow(model_mask, cmap="gray", alpha=0.25)

    fig = plt.figure(3, figsize=(10, 7))
    plt.imshow(probe_img)
    plt.imshow(probe_mask, cmap="gray", alpha=0.25)
    plt.show()

# plot_images_overlay()
def plot_images_scores(model, probe, title):
    # code for displaying multiple images in one figure

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(model, interpolation="nearest")
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(probe, interpolation="nearest")
    plt.axis('off')
    plt.title("Second")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(visualize_xor(model, probe), interpolation="nearest")
    plt.axis('off')
    plt.title("XOR")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(visualize_diff(model, probe), interpolation="nearest")
    plt.axis('off')
    plt.title("Difference")

    score = compute_hamming_dist(model, probe)

    plt.suptitle(title + " - " + str(score))
    plt.show()

def affine_transform(img, a, b, c, d, e, f):
    shape = img.shape
    transformed = np.zeros(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if img[y, x] == 1:
                x_t = round(a * x + b * y + c)
                y_t = round(d * x + e * y + f)
                if x_t >= 0 and x_t < shape[1] and y_t >= 0 and y_t < shape[0]:
                    transformed[y_t, x_t] = 1.0

    return transformed

def search_affine_transform(model_key_points, probe_key_points):
    print(model_key_points, probe_key_points)

    x1 = [x for (x, _) in probe_key_points]
    x2 = [x for (x, _) in model_key_points]

    y1 = [y for (_, y) in probe_key_points]
    y2 = [y for (_, y) in model_key_points]
    coefficients = np.column_stack((x1, y1, np.ones_like(x1)))
    print(coefficients)
    abc_, _, _, _ = np.linalg.lstsq(coefficients, x2, rcond=None)
    def_, _, _, _ = np.linalg.lstsq(coefficients, y2, rcond=None)
    print(abc_)

    print(def_)
    return list(abc_), list(def_)

# plot_images(model, probe, "No Alignment")

# first try miura matching:
model_miura, a, probe_miura, b = postprocess(model, model, probe, probe, "miura_matching")
plot_images_scores(model_miura, probe_miura, "Miura Matching Algorithm")

# probe_aff = affine_transform(probe, 1, 0, 5, 0, 0.95, -10)
# plot_images(model, probe_aff, "Affine Transform")

##################### Select key point coordinates
num_key_points = 10
shape = model.shape
model_key_points = []
probe_key_points = []
run = [False]

def main():
    print("running main ...")

    abc_, def_ = search_affine_transform(model_key_points, probe_key_points)
    probe_aff = affine_transform(probe, *abc_, *def_)
    plot_images_scores(model, probe_aff, "Custom Affine Transform")

def probe_mouse_event(event):
    if len(probe_key_points) < num_key_points:
        x = round(event.xdata)
        x = min(max(x, 0), shape[1] - 1)
        y = round(event.ydata)
        y = min(max(y, 0), shape[0] - 1)

        print('x: {} and y: {}'.format(x, y))
        probe_key_points.append((x, y))
    else:
        pass # simply ignore

    if len(probe_key_points) == num_key_points and len(model_key_points) == num_key_points and not run[0]:
        run[0] = True
        main()

def model_mouse_event(event):

    if len(model_key_points) < num_key_points:
        x = round(event.xdata)
        x = min(max(x, 0), shape[1] - 1)
        y = round(event.ydata)
        y = min(max(y, 0), shape[0] - 1)

        print('x: {} and y: {}'.format(x, y))
        model_key_points.append((x, y))
    else:
        pass # simply ignore

    if len(probe_key_points) == num_key_points and len(model_key_points) == num_key_points and not run[0]:
        run[0] = True
        main()

# plot_images_overlay()

# fig = plt.figure(1)
# cid = fig.canvas.mpl_connect('button_press_event', model_mouse_event)
# plt.imshow(model)
# fig2 = plt.figure(2)
# cid2 = fig2.canvas.mpl_connect('button_press_event', probe_mouse_event)
# plt.imshow(probe)
# plt.show()