import time

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import matplotlib as mpl

import alignment_evaluation
from alignment_evaluation import *
# importing the module
import cv2
mpl.use('WebAgg')


# load images
model_path = 'dataset_i/maximum_curvature/5_left_ring_1_cam1.npy'
probe_path = 'dataset_i/maximum_curvature/5_left_ring_2_cam1.npy'
model = np.load(model_path)
probe = np.load(probe_path)

def visualize_diff(a, b):
    return 2 * a + b

def visualize_xor(a, b):
    return np.bitwise_xor(a.astype(int), b.astype(int))

def plot_images(model, probe, title):
    # code for displaying multiple images in one figure

    # import libraries
    import cv2
    from matplotlib import pyplot as plt

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

    x1 = [x for (x, _) in model_key_points]
    x2 = [x for (x, _) in probe_key_points]

    y1 = [y for (_, y) in model_key_points]
    y2 = [y for (_, y) in probe_key_points]

    m1 = [[]]


# plot_images(model, probe, "No Alignment")

# first try miura matching:
# model_miura, a, probe_miura, b = postprocess(model, model, probe, probe, "miura_matching")
# plot_images(model_miura, probe_miura, "Miura Matching Algorithm")

# probe_aff = affine_transform(probe, 1, 0, 5, 0, 0.95, -10)
# plot_images(model, probe_aff, "Affine Transform")


##################### Select key point coordinates
num_key_points = 3
shape = model.shape
model_key_points = []
probe_key_points = []
run = [False]

def main():
    print("running main ...")
    search_affine_transform(model_key_points, probe_key_points)


def probe_mouse_event(event):
    if len(probe_key_points) < num_key_points:
        x = round(event.xdata)
        x = min(max(x, 0), shape[1] - 1)
        y = round(event.ydata)
        y = min(max(y, 0), shape[0] - 1)

        print('x: {} and y: {}'.format(x, y))
        probe_key_points.append((x, y))
    elif len(probe_key_points) == num_key_points and len(model_key_points) == num_key_points and not run[0]:
        run[0] = True
        main()
    else:
        pass # simply ignore

def model_mouse_event(event):

    if len(model_key_points) < num_key_points:
        x = round(event.xdata)
        x = min(max(x, 0), shape[1] - 1)
        y = round(event.ydata)
        y = min(max(y, 0), shape[0] - 1)

        print('x: {} and y: {}'.format(x, y))
        model_key_points.append((x, y))
    elif len(probe_key_points) == num_key_points and len(model_key_points) == num_key_points and not run[0]:
        run[0] = True
        main()
    else:
        pass # simply ignore

fig = plt.figure(1)
cid = fig.canvas.mpl_connect('button_press_event', model_mouse_event)
plt.imshow(model)
fig2 = plt.figure(2)
cid2 = fig2.canvas.mpl_connect('button_press_event', probe_mouse_event)
plt.imshow(probe)
plt.show()
