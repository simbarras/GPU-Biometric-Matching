import matplotlib.pyplot as plt

from .extraction_pipeline import *

def compute_single_distance(model, probe, distance_function):
    if distance_function == "hamming_dist" or distance_function == "hamming_distance":
        return compute_hamming_dist(model, probe)
    elif distance_function == "random_subsampling_dist":
        return compute_random_subsampling_dist(model, probe)
    elif distance_function == "skeleton_hd":
        return compute_skeleton_hd(model, probe)
    elif distance_function == "miura_distance" or distance_function == "miura_dist":
        return compute_miura_distance(model, probe)
    elif distance_function == "always_perfect":
        return 0
    else:
        raise NotImplementedError()

def compute_miura_distance(model, probe):
    aandb = np.bitwise_and(model.astype(int), probe.astype(int))
    dist = 1 - (np.count_nonzero(aandb) / (np.count_nonzero(model) + np.count_nonzero(probe)))
    return dist

def compute_skeleton_hd(a, b, min_area=30):
    """
    Created by Simon, description in project Fuzzy Extraction for Finger Veins.
    Note: this is actually a similarity measure and not a distance function.
    """
    axorb = np.bitwise_and(a.astype(int), b.astype(int))
    axorb = skeletonize(axorb)
    blobs, labnbr = si.label(axorb, structure = np.array([[1, 1, 1],
                                                        [1, 1, 1],
                                                         [1, 1, 1]]))
    pixels = blobs.ravel()
    areas = np.bincount(pixels)[1:]
    kept_labels = np.argwhere(areas > min_area) + 1
    axorb = np.isin(blobs, kept_labels).astype(dtype="uint16")

    ham_dist = np.count_nonzero(axorb == 1) / axorb.size
    return round(ham_dist, 6)

def compute_hamming_dist(a, b):
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))
    nr_of_ones = np.count_nonzero(axorb == 1)
    ham_dist = nr_of_ones / axorb.size
    return round(ham_dist, 6)

def compute_random_subsampling_dist(a, b):
    """
    Created by Simon, description in project Fuzzy Extraction for Finger Veins, chapter 3.
    """
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))

    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)
    nr_of_ones_match = nr_of_ones_a + nr_of_ones_b - np.count_nonzero(axorb == 1)

    # distance is the probability that a single pixel is not covered by the model
    dist = 1 - nr_of_ones_match / (nr_of_ones_a * nr_of_ones_b)
    return dist