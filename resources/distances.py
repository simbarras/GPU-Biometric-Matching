import matplotlib.pyplot as plt

from .extraction_pipeline import *

def compute_single_distance(model, probe, distance_function):
    if distance_function == "hamming_dist" or distance_function == "hamming_distance":
        return compute_hamming_dist(model, probe)
    elif distance_function == "random_subsampling_dist":
        return compute_random_subsampling_dist(model, probe)
    elif distance_function == "skeleton_hd":
        return compute_skeleton_hd(model, probe)
    elif distance_function == "overlap_distance":
        return compute_overlap_dist(model, probe)
    elif distance_function == "miura_distance" or distance_function == "miura_dist":
        return compute_miura_distance(model, probe)
    elif distance_function == "always_perfect":
        return 0
    else:
        raise NotImplementedError()

def compute_overlap_dist(model, probe):
    # model = shift(model, 10, 10)
    #plt.imshow(model + 2 * probe)
    dist = np.sum(np.bitwise_and(model.astype(int), probe.astype(int)))
    #plt.suptitle(dist)
    #plt.show()
    return dist

def compute_miura_distance(model, probe):
    aandb = np.bitwise_and(model.astype(int), probe.astype(int))

    #plt.imshow(model + 2 * probe)
    dist = 1 - (np.count_nonzero(aandb) / (np.count_nonzero(model) + np.count_nonzero(probe)))
    #plt.suptitle(dist)
    #plt.show()
    return dist

def compute_skeleton_hd(a, b, min_area=30):
    axorb = np.bitwise_and(a.astype(int), b.astype(int))
    axorb = skeletonize(axorb)

    # remove parts too small
    # remove noise
    blobs, labnbr = si.label(axorb, structure = np.array([[1, 1, 1],
                                                        [1, 1, 1],
                                                         [1, 1, 1]]))
    pixels = blobs.ravel()
    areas = np.bincount(pixels)[1:]
    kept_labels = np.argwhere(areas > min_area) + 1
    axorb = np.isin(blobs, kept_labels).astype(dtype="uint16")

    ham_dist = np.count_nonzero(axorb == 1) / axorb.size

    #if ham_dist < 0.004:
    #plt.imshow(axorb)

    #plt.imshow(axorb + 2 * a + b)
    #plt.suptitle(str(ham_dist))
    #plt.show()
    return round(ham_dist, 6)

def compute_hamming_dist(a, b):
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))

    ###### VISUALIZE HAMMING DISTANCE
    #plt.imshow(2 * a + b)
    #plt.show()

    nr_of_ones = np.count_nonzero(axorb == 1)
    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)

    ham_dist = nr_of_ones / axorb.size
    return round(ham_dist, 6)

def compute_random_subsampling_dist(a, b):

    axorb = np.bitwise_xor(a.astype(int), b.astype(int))

    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)
    nr_of_ones_match = nr_of_ones_a + nr_of_ones_b - np.count_nonzero(axorb == 1)

    # distance is the probability that a single pixel is not covered by the model
    dist = 1 - nr_of_ones_match / (nr_of_ones_a * nr_of_ones_b)

    ############ Visualizations
    #plt.imshow(2 * a + b)
    #plt.imshow(mask_a * 2 + mask_b, cmap="gray", alpha=0.5)
    #plt.savefig("inspection/" + pop + "_" + img_name_a + "_" + img_name_b + ".png")
    #plt.show()

    return dist