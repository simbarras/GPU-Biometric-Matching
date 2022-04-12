from .extraction_pipeline import *

def compute_hamming_dist(a, b):
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))

    ###### VISUALIZE HAMMING DISTANCE
    # plt.imshow(2 * a + b)
    # plt.show()
    # plt.imshow(b)
    # plt.show()

    nr_of_ones = np.count_nonzero(axorb == 1)
    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)

    ham_dist = nr_of_ones / axorb.size
    return round(ham_dist, 6)
