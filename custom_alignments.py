import alignment_evaluation
from alignment_evaluation import *


def plot_images(model, probe):
    plt.imshow(model)
    plt.show()
    plt.imshow(probe)
    plt.show()


def visualize_diff(model, probe):
    plt.imshow(model)
    plt.show()
    plt.imshow(probe)
    plt.show()


# load images
model = np.load('dataset_i/maximum_curvature/3_left_index_1_cam1.npy')
probe = np.load('dataset_i/maximum_curvature/3_left_index_2_cam1.npy')


# first try miura matching:
model_miura, a, probe_miura, b = postprocess(model, model, probe, probe, "miura_matching")
