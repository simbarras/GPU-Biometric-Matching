import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as sd
from resources import shift, fingerfocus, extract_features, shift_to_CoM, miurascore
from PIL import Image


def compute_hamming_dist(a,b):
    axorb = np.bitwise_xor(a.astype(int),b.astype(int))
# just to check we're not results for computing something off
#     plt.imshow(a)
#     plt.show()
#     plt.imshow(b)
#     plt.show()
#     plt.imshow(axorb)
#     plt.show()
    
    nr_of_ones = np.count_nonzero(axorb == 1)
    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)
    print("nr of 1s in xor: ",nr_of_ones, "in a:", nr_of_ones_a, "in b:", nr_of_ones_b)

    ham_dist = nr_of_ones/axorb.size
    print("axorb size: ", axorb.size, "ham_dist:", ham_dist)
    # alternative, 1 liner:
    # ham_dist = sd.hamming(a.flatten(), b.flatten())
    return (round(ham_dist,6),nr_of_ones,np.count_nonzero(a.astype(int) == 1), np.count_nonzero(b.astype(int) == 1))


'''
the main:
to test method_i one needs to modify the biocore preprocess function so that it calls the function tested
add this line according to the method tested
# data, mask = huang_normalization(data, mask, False, False) - method 2
# data, mask = huang_normalization(data, mask, True, False) - method 3
# data, mask = align_leftmost_edge(data,mask) - method 4
# data, mask = huang_normalization(data, mask, False, True) - method 5
method 1 shift_to_CoM was applied on the 0/1 img after the feature extraction:
# W = shift_to_CoM(W)
# W_tilde_same = shift_to_CoM(W_tilde_same)
for none: none of the functions above needs to be called
for optimal: none of these needs to be called, instead
# ch = 30
# cw = 90
# score, t0, s0 = miurascore(W, W_tilde_same, retmax=True)
# W_tilde_same = shift(W_tilde_same,t0-ch,s0-cw)
'''

hd_res = []
for user in ['5','3']:
    for finger in ['middle','ring', 'index']:
        for cam in ['cam1','cam2']:
            for lr in ['left','right']:
                img_trial1 = "dataset_i/" + user + "_" + lr + "_" + finger + "_1_" + cam + ".png"
                img_trial2 = "dataset_i/" + user + "_" + lr + "_"  + finger + "_2_" + cam + ".png"
                if img_trial1 != "dataset_i/5_left_middle_1_cam1.png" and img_trial1 != "dataset_i/5_left_middle_1_cam2.png" and img_trial1 != "dataset_i/5_left_index_1_cam1.png" and img_trial1 != "dataset_i/5_left_index_1_cam2.png":
                    print("Load and extract W", img_trial1)
                    W = Image.open(img_trial1)
                    W = np.asarray(W)
                    W, mask = fingerfocus(W, roi=(40, 190, 10, 360))
                    W, mask = extract_features(W, mask)
                    # W = shift_to_CoM(W)

                    print("Load and extract W_tilde_same", img_trial2)
                    W_tilde_same = Image.open(img_trial2)
                    W_tilde_same = np.asarray(W_tilde_same)
                    W_tilde_same, mask_tilde = fingerfocus(W_tilde_same, roi = (40, 190, 10, 360))
                    W_tilde_same, mask_tilde = extract_features(W_tilde_same, mask_tilde)
                    # W_tilde_same = shift_to_CoM(W_tilde_same)


                    plt.imshow(W)
                    plt.savefig("extracted_features_i/"+ user + "_" + lr + "_" + finger + "_1_" + cam + "_extracted.png")
                    plt.imshow(W_tilde_same)
                    plt.savefig("extracted_features_i/"+ user + "_" + lr + "_" + finger + "_2_" + cam + "_extracted.png")

                    # compute the optimal params; comment out self-made functions in preprocess (biocore.py)
                    # compute hamming distance between the two W, W_tilde_same
                    # ch = 30
                    # cw = 90
                    # score, t0, s0 = miurascore(W, W_tilde_same, retmax=True)
                    # print("miurascore:", score, t0, s0)
                    # hd_res.append(compute_hamming_dist(W,shift(W_tilde_same,t0-ch,s0-cw)))
                    hd_res.append(compute_hamming_dist(W,W_tilde_same))
print(hd_res)
