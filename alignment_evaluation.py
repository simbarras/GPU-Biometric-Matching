import itertools
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as sd
from resources import shift, fingerfocus, extract_features, shift_to_CoM, miurascore
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
from IPython.display import display
from collections.abc import Iterable

# source: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def prod_index(cartesian_params, cartesian_names, comb_param=None, comb_names=None, comb_param_pos=0):
    """
    Returns product index of specified parameters. If parameter is None, then same value as previous value will be used.
    Further, it prevents the output from having symmetric distances (same measurement not done twice with other value as
    model and previous model as probe.

    @param cartesian_params : Hello
    """
    comb = []
    if comb_param is not None:
        for i in itertools.combinations(comb_param, 2):
            comb.append(i)
        cartesian_params = cartesian_params[:comb_param_pos] + [comb] + cartesian_params[comb_param_pos:]

    print(cartesian_params)
    cartesian_tuples = []
    for i in itertools.product(*cartesian_params):
        tpl_list = []
        prev = None
        for v in list(i):
            if v is None:
                v = prev
            tpl_list.append(v)
            prev = v

        tpl = tuple(tpl_list)
        cartesian_tuples.append(tuple(flatten(tpl)))

    # TODO: replace product index with this construction -> implement logic to determine comb_param and cartesian_params



def dataframe_generator(spec=None, out=None):
    """ Generates a dataframe with candidate tuples. For each tuple the indicated score will be computed.
    The dataframe is then stored as a CSV in the "experiments" folder. Caches extracted features if not cached yet.
    """

    # specification defaults:
    if spec is None:
        spec = {
            "dataset_id": ['i'],                        # datasets numbered with roman numbers
            "score_function": ['hamming_dist'],         # [hamming_dist, ...]
            "feature_extractor": ['max_curvature'],     # [max_curvature, wide_line, repeated_line]
            "alignment": ['id'],                        # [id, cm, ...]
            "side_m": ['left', 'right'],                # [left, right]
            "side_p": ['left', 'right'],                             # none -> same as m
            "finger_m": ['index', 'ring', 'middle', 'little', 'thumb'],    # [little, ring, middle, index, thumb]
            "finger_p": ['index', 'ring', 'middle', 'little', 'thumb'],                           # none -> same as m
            "camera_m": [1,2],                          # [1, 2]
            "camera_p": [1,2],                           # [none] -> same as m
            "trial_m": [],                              # [] -> all samples, otherwise as specified
            "trial_p": [],                              # same as above (note none does not work here)
            "id_m": [],                                 # [] -> all samples, otherwise as specified
            "id_p": [],                               # [none] -> same as m, otherwise as above.
        }

    # load dataset:
    dataset_path = "dataset_" + spec["dataset_id"][0]
    dataset = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]


    ######################
    # INDEX CREATION     #
    ######################


    # replace "None" by actual parameters (note, overestimation, values that don't exist get removed later):
    trials = set()
    ids = set()
    for img_path in dataset:
        words = img_path.split("_")
        trials.add(int(words[3]))
        ids.add(int(words[0]))

    if not spec["trial_m"] is None and not spec["trial_m"]:
        spec["trial_m"] = list(trials)
    if not spec["trial_p"] is None and not spec["trial_p"]:
        spec["trial_p"] = list(trials)
    if not spec["id_m"] is None and not spec["id_m"]:
        spec["id_m"] = list(ids)
    if not spec["id_p"] is None and not spec["id_p"]:
        spec["id_p"] = list(ids)

    # if finger_p is none or camera_p is none, the respective columns get removed.
    original_columns = list(spec.keys())
    if spec["finger_p"] is None:
        spec.pop("finger_p")
    if spec["camera_p"] is None:
        spec.pop("camera_p")
    if spec["side_p"] is None:
        spec.pop("side_p")
    if spec["id_p"] is None:
        spec.pop("id_p")

    # create index:
    columns = spec.keys()
    params = []
    for c in columns:
        params.append(spec[c])
    index = pd.MultiIndex.from_product(params, names=list(columns))
    # TODO: improve product index s.t. same tuple doesn't exist twice (remove tuples where only trials are switched)


    # add back rows if they were removed (please notice the beauty of this code):
    if "camera_p" not in columns or "finger_p" not in columns or "side_p" not in columns or "id_p" not in columns:
        tuples = []
        if "finger_p" not in columns and "camera_p" in columns and "side_p" in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, c_m, c_p, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_m, c_m, c_p, t_m, t_p, i_m, i_p))
        elif "camera_p" not in columns and "finger_p" in columns and "side_p" in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, c_m, t_m, t_p, i_m, i_p))
        elif "side_p" not in columns and "camera_p" in columns and "finger_p" in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, f_m, f_p, c_m, c_p, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_p, c_m, c_p, t_m, t_p, i_m, i_p))
        elif "side_p" not in columns and "camera_p" not in columns and "finger_p" in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, f_m, f_p, c_m, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_p, c_m, c_m, t_m, t_p, i_m, i_p))
        elif "side_p" not in columns and "camera_p" in columns and "finger_p" not in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, f_m, c_m, c_p, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_m, c_m, c_p, t_m, t_p, i_m, i_p))
        elif "side_p" in columns and "camera_p" not in columns and "finger_p" not in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, c_m, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_m, c_m, c_m, t_m, t_p, i_m, i_p))
        elif "side_p" not in columns and "camera_p" not in columns and "finger_p" not in columns and "id_p" in columns:
            for i, (d_id, score, fe, al, s_m, f_m, c_m, t_m, t_p, i_m, i_p) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_m, c_m, c_m, t_m, t_p, i_m, i_p))
        elif "finger_p" in columns and "camera_p" in columns and "side_p" in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, c_p, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, c_p, t_m, t_p, i_m, i_m))
        elif "finger_p" not in columns and "camera_p" in columns and "side_p" in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, c_m, c_p, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_m, c_m, c_p, t_m, t_p, i_m, i_m))
        elif "camera_p" not in columns and "finger_p" in columns and "side_p" in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_p, c_m, c_m, t_m, t_p, i_m, i_m))
        elif "side_p" not in columns and "camera_p" in columns and "finger_p" in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, f_m, f_p, c_m, c_p, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_p, c_m, c_p, t_m, t_p, i_m, i_m))
        elif "side_p" not in columns and "camera_p" not in columns and "finger_p" in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, f_m, f_p, c_m, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_p, c_m, c_m, t_m, t_p, i_m, i_m))
        elif "side_p" not in columns and "camera_p" in columns and "finger_p" not in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, f_m, c_m, c_p, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_m, c_m, c_p, t_m, t_p, i_m, i_m))
        elif "side_p" in columns and "camera_p" not in columns and "finger_p" not in columns and "id_p" not in columns:
            for i, (d_id, score, fe, al, s_m, s_p, f_m, c_m, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_p, f_m, f_m, c_m, c_m, t_m, t_p, i_m, i_m))
        else: # all are missing
            for i, (d_id, score, fe, al, s_m, f_m, c_m, t_m, t_p, i_m) in enumerate(index):
                tuples.append((d_id, score, fe, al, s_m, s_m, f_m, f_m, c_m, c_m, t_m, t_p, i_m, i_m))
        index = pd.MultiIndex.from_tuples(tuples, names=list(original_columns))

    def tuple_to_filename(tpl):
        img_m = str(tpl[12]) + "_" + tpl[4] + "_" + tpl[6] + "_" + str(tpl[10]) + "_cam" + str(tpl[8]) + ".png"
        img_p = str(tpl[13]) + "_" + tpl[5] + "_" + tpl[7] + "_" + str(tpl[11]) + "_cam" + str(tpl[9]) + ".png"
        return img_m, img_p

    # delete obsolete values from index and add back missing columns (where it was none):
    delete_ids = []
    dataset_membership = set(dataset) # more efficient datastructure to check membership
    for i, tpl in enumerate(index):
        # delete tuple from index if image does not exist or same image would be compared
        if tpl[4] == tpl[5] and tpl[6] == tpl[7] and tpl[8] == tpl[9] and tpl[10] == tpl[11] and tpl[12] == tpl[13]:
            delete_ids.append(i)
            continue

        # check if img exists in dataset for both that are being compared
        img_m, img_p = tuple_to_filename(tpl)

        if img_m not in dataset_membership or img_p not in dataset_membership:
            delete_ids.append(i)
            continue

    index = index.delete(delete_ids)


    ######################
    # Create Dataframe   #
    ######################

    # create pandas dataframe:
    df = pd.DataFrame(index=index, columns=["score"])
    display(df)

# dataframe_generator()
prod_index([[1, 2, 3, 4, 5], ["a", "b", "c"], [None]], ["rhabarber"], [1, 2, 3, 4, 5], ["ferdinand"], 3)

def compute_hamming_dist(a, b):
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))
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
    print("nr of 1s in xor: ", nr_of_ones, "in a:", nr_of_ones_a, "in b:", nr_of_ones_b)

    ham_dist = nr_of_ones / axorb.size
    print("axorb size: ", axorb.size, "ham_dist:", ham_dist)
    # alternative, 1 liner:
    # ham_dist = sd.hamming(a.flatten(), b.flatten())
    return (round(ham_dist, 6), nr_of_ones, np.count_nonzero(a.astype(int) == 1), np.count_nonzero(b.astype(int) == 1))


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

# hd_res = []
# for user in ['5','3']:
#     for finger in ['middle','ring', 'index']:
#         for cam in ['cam1','cam2']:
#             for lr in ['left','right']:
#                 img_trial1 = "dataset_i/" + user + "_" + lr + "_" + finger + "_1_" + cam + ".png"
#                 img_trial2 = "dataset_i/" + user + "_" + lr + "_"  + finger + "_2_" + cam + ".png"
#                 if img_trial1 != "dataset_i/5_left_middle_1_cam1.png" and img_trial1 != "dataset_i/5_left_middle_1_cam2.png" and img_trial1 != "dataset_i/5_left_index_1_cam1.png" and img_trial1 != "dataset_i/5_left_index_1_cam2.png":
#                     print("Load and extract W", img_trial1)
#                     W = Image.open(img_trial1)
#                     W = np.asarray(W)
#                     W, mask = fingerfocus(W, roi=(40, 190, 10, 360))
#                     W, mask = extract_features(W, mask)
#                     # W = shift_to_CoM(W)
#
#                     print("Load and extract W_tilde_same", img_trial2)
#                     W_tilde_same = Image.open(img_trial2)
#                     W_tilde_same = np.asarray(W_tilde_same)
#                     W_tilde_same, mask_tilde = fingerfocus(W_tilde_same, roi = (40, 190, 10, 360))
#                     W_tilde_same, mask_tilde = extract_features(W_tilde_same, mask_tilde)
#                     # W_tilde_same = shift_to_CoM(W_tilde_same)
#
#                     # save extracted features to directory ###############################
#                     # plt.imshow(W)
#                     # plt.savefig("fe_max_curvature_i/"+ user + "_" + lr + "_" + finger + "_1_" + cam + "_extracted.png")
#                     # plt.imshow(W_tilde_same)
#                     # plt.savefig("fe_max_curvature_i/"+ user + "_" + lr + "_" + finger + "_2_" + cam + "_extracted.png")
#
#                     # CUSTOM FUNCTIONS ##########################################
#                     # compute the optimal params; comment out self-made functions in preprocess (biocore.py)
#                     # compute hamming distance between the two W, W_tilde_same
#                     # ch = 30
#                     # cw = 90
#                     # score, t0, s0 = miurascore(W, W_tilde_same, retmax=True)
#                     # print("miurascore:", score, t0, s0)
#                     # hd_res.append(compute_hamming_dist(W,shift(W_tilde_same,t0-ch,s0-cw)))
#                     hd_res.append(compute_hamming_dist(W,W_tilde_same))
# print(hd_res)
