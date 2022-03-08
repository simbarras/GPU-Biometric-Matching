import itertools
import json

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


######################## globals
experiment_dir_pref = "experiments/experiment_"
dataset_dir_pref = "dataset_"


# source: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    """
    Flattens an arbitrarily nested list or tuple
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def prod_index(cartesian_params, comb_param_pos=0):
    """
    Returns product index of specified parameters. If parameter is None, then same value as previous value will be used.
    Further, it prevents the output from having symmetric distances (same measurement not done twice with other value as
    model and previous model as probe.
    """


    comb = []
    if comb_param_pos is not None:
        comb_param = cartesian_params[comb_param_pos]
        cartesian_params = cartesian_params[: comb_param_pos] + cartesian_params[comb_param_pos + 2:]

        for i in itertools.combinations(comb_param, 2):
            comb.append(i)
        cartesian_params = cartesian_params[:comb_param_pos] + [comb] + cartesian_params[comb_param_pos:]
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
    return cartesian_tuples

def tuple_to_filename(tpl, idx):
    img_m = str(tpl[idx["id_m"]]) + "_" + tpl[idx["side_m"]] + "_" + tpl[idx["finger_m"]] + "_" + str(tpl[idx["trial_m"]]) + "_cam" + str(tpl[idx["camera_m"]]) + ".png"
    img_p = str(tpl[idx["id_p"]]) + "_" + tpl[idx["side_p"]] + "_" + tpl[idx["finger_p"]] + "_" + str(tpl[idx["trial_p"]]) + "_cam" + str(tpl[idx["camera_p"]]) + ".png"
    return img_m, img_p

def post_filter_index(idx, index, dataset):
    # delete obsolete values from index and add back missing columns (where it was none):
    delete_ids = []
    dataset_membership = set(dataset)  # more efficient datastructure to check membership
    for i, tpl in enumerate(index):
        # delete tuple from index if image does not exist or same image would be compared
        if tpl[idx["id_m"]] == tpl[idx["id_p"]] \
                and tpl[idx["side_m"]] == tpl[idx["side_p"]] \
                and tpl[idx["finger_m"]] == tpl[idx["finger_p"]] \
                and tpl[idx["trial_m"]] == tpl[idx["trial_p"]] \
                and tpl[idx["camera_m"]] == tpl[idx["camera_p"]]:
            delete_ids.append(i)
            continue

        # check if img exists in dataset for both that are being compared
        img_m, img_p = tuple_to_filename(tpl, idx)

        if img_m not in dataset_membership or img_p not in dataset_membership:
            delete_ids.append(i)
            continue
    return index.delete(delete_ids)

def dataframe_generator(spec=None, idx=None, combination_parameter_pos=None, out=None):
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
            "id_m": [],                                 # [] -> all samples, otherwise as specified
            "id_p": [None],                               # [none] -> same as m, otherwise as above.
            "side_m": ['left', 'right'],                # [left, right]
            "side_p": [None],                             # none -> same as m
            "finger_m": ['index', 'ring', 'middle', 'little', 'thumb'],    # [little, ring, middle, index, thumb]
            "finger_p": [None],                           # none -> same as m
            "trial_m": [],                              # [] -> all samples, otherwise as specified
            "trial_p": [],                              # same as above (note none does not work here)
            "camera_m": [1,2],                          # [1, 2]
            "camera_p": [None],                           # [none] -> same as m
        }

    if idx is None:
        idx = {
            "dataset_id": 0,
            "score_function": 1,
            "feature_extractor": 2,
            "alignment": 3,
            "id_m": 4,
            "id_p": 5,
            "side_m": 6,
            "side_p": 7,
            "finger_m": 8,
            "finger_p": 9,
            "trial_m": 10,
            "trial_p": 11,
            "camera_m": 12,
            "camera_p": 13
        }

    # load dataset:
    dataset_path = dataset_dir_pref + spec["dataset_id"][0]
    dataset = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    ######################
    # INDEX CREATION     #
    ######################


    # replace "[]" by actual parameters (note, overestimation, values that don't exist get removed later):
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

    # create index:
    columns = spec.keys()
    params = []
    for c in columns:
        params.append(spec[c])
    foo = prod_index(params, combination_parameter_pos)
    index = pd.MultiIndex.from_tuples(foo, names=list(columns))
    index = post_filter_index(idx, index, dataset)

    ######################
    # Create Dataframe   #
    ######################

    # create pandas dataframe:
    df = pd.DataFrame(index=index, columns=["score"])
    if out is not None:
        df.to_csv(out)
    return df

def compute_hamming_dist(a, b):
    axorb = np.bitwise_xor(a.astype(int), b.astype(int))
    # just to check we're not results for computing something off
    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
    plt.imshow(axorb)
    plt.show()

    nr_of_ones = np.count_nonzero(axorb == 1)
    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)
    print("nr of 1s in xor: ", nr_of_ones, "in a:", nr_of_ones_a, "in b:", nr_of_ones_b)

    ham_dist = nr_of_ones / axorb.size
    print("axorb size: ", axorb.size, "ham_dist:", ham_dist)
    # alternative, 1 liner:
    # ham_dist = sd.hamming(a.flatten(), b.flatten())
    return (round(ham_dist, 6), nr_of_ones, np.count_nonzero(a.astype(int) == 1), np.count_nonzero(b.astype(int) == 1))

def calculate_scores(idx, dataset_path, in_path=None, out_path=None, df=None):
    if in_path is not None:
        df = pd.read_csv(in_path)
    for row in df.itertuples():
        row = list(row)
        i = row[0]
        tpl = row[1:]
        model_path, probe_path = tuple_to_filename(tpl, idx)
        model_path = dataset_path + model_path
        probe_path = dataset_path + probe_path


        ###################################################################### DALIA
        print("Load and extract W", model_path)
        W = Image.open(model_path)
        W = np.asarray(W)
        W, mask = fingerfocus(W, roi=(40, 190, 10, 360))
        W, mask = extract_features(W, mask)
        # W = shift_to_CoM(W)

        print("Load and extract W_tilde_same", probe_path)
        W_tilde_same = Image.open(probe_path)
        W_tilde_same = np.asarray(W_tilde_same)
        W_tilde_same, mask_tilde = fingerfocus(W_tilde_same, roi = (40, 190, 10, 360))
        W_tilde_same, mask_tilde = extract_features(W_tilde_same, mask_tilde)
        # W_tilde_same = shift_to_CoM(W_tilde_same)

        # save extracted features to directory ###############################
        # plt.imshow(W)
        # plt.savefig("fe_max_curvature_i/"+ user + "_" + lr + "_" + finger + "_1_" + cam + "_extracted.png")
        # plt.imshow(W_tilde_same)
        # plt.savefig("fe_max_curvature_i/"+ user + "_" + lr + "_" + finger + "_2_" + cam + "_extracted.png")

        # CUSTOM FUNCTIONS ##########################################
        # compute the optimal params; comment out self-made functions in preprocess (biocore.py)
        # compute hamming distance between the two W, W_tilde_same
        # ch = 30
        # cw = 90
        # score, t0, s0 = miurascore(W, W_tilde_same, retmax=True)
        # print("miurascore:", score, t0, s0)
        # hd_res.append(compute_hamming_dist(W,shift(W_tilde_same,t0-ch,s0-cw)))
        print(compute_hamming_dist(W, W_tilde_same))
        # hd_res.append(compute_hamming_dist(W,W_tilde_same))

        ###################################################################### DALIA

        df.at[i, "score"] = 33

def run_experiment(experiment_id="i"):
    # load specification and idx
    f = open(experiment_dir_pref + experiment_id + "/spec.json")
    experiment_spec = json.load(f)
    f.close()
    spec = experiment_spec["spec"]
    idx = experiment_spec["idx"]
    combination_param_pos = experiment_spec["combination_param_pos"]

    # generate empty dataset if it does not exist yet and store it
    out_path = experiment_dir_pref + experiment_id + "/setup.csv"
    df = None
    if not isfile(out_path):
        df = dataframe_generator(spec=spec, idx=idx, combination_parameter_pos=combination_param_pos, out=out_path)

    # calculate scores of dataframe
    scores_out_path = experiment_dir_pref + experiment_id + "/results.csv"
    dataset_path = dataset_dir_pref + spec["dataset_id"][0] + "/"
    calculate_scores(idx, dataset_path=dataset_path, in_path=out_path, out_path=scores_out_path, df=df)


run_experiment("i")