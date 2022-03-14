import itertools
import json
import os
import roman
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as sd
from resources import shift, fingerfocus, extract_features, shift_to_CoM, miurascore, postprocess
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

def num_to_roman(n):
    return roman.toRoman(n).lower()

def tuple_to_filename(tpl, idx, suffix):
        img_m = str(tpl[idx["id_m"]]) + "_" + tpl[idx["side_m"]] + "_" + tpl[idx["finger_m"]] + "_" + str(
            tpl[idx["trial_m"]]) + "_cam" + str(tpl[idx["camera_m"]]) + suffix
        img_p = str(tpl[idx["id_p"]]) + "_" + tpl[idx["side_p"]] + "_" + tpl[idx["finger_p"]] + "_" + str(
            tpl[idx["trial_p"]]) + "_cam" + str(tpl[idx["camera_p"]]) + suffix
        return img_m, img_p

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

    # helper functions:
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
            img_m, img_p = tuple_to_filename(tpl, idx, suffix='.jpg')

            if img_m not in dataset_membership or img_p not in dataset_membership:
                delete_ids.append(i)
                continue
        return index.delete(delete_ids)

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
    #plt.imshow(a)
    #plt.show()
    #plt.imshow(b)
    #plt.show()
    #plt.imshow(axorb)
    #plt.show()

    nr_of_ones = np.count_nonzero(axorb == 1)
    nr_of_ones_a = np.count_nonzero(a == 1)
    nr_of_ones_b = np.count_nonzero(b == 1)
    print("nr of 1s in xor: ", nr_of_ones, "in a:", nr_of_ones_a, "in b:", nr_of_ones_b)

    ham_dist = nr_of_ones / axorb.size
    print("axorb size: ", axorb.size, "ham_dist:", ham_dist)
    # alternative, 1 liner:
    # ham_dist = sd.hamming(a.flatten(), b.flatten())
    return (round(ham_dist, 6), nr_of_ones, np.count_nonzero(a.astype(int) == 1), np.count_nonzero(b.astype(int) == 1))

def load_and_extract(img_path, out_path, feature_extractor=None):
    print("Load and extract W", img_path)
    W = Image.open(img_path)
    W = np.asarray(W)
    W, mask = fingerfocus(W, roi=(40, 190, 10, 360))
    # TODO: use feature_extractor as a selector.
    W, mask = extract_features(W, mask)
    np.save(out_path + "_mask", mask)
    np.save(out_path, W)

def preprocess_alignment_method(alignment_method):
    """
    @param alignment_method: alignment method used to transform image
    @return: if alignment method is used before feature extraction, it returns the same name, otherwise it returns
    an empty string. This is used to get the correct file name of the cached extracted feature.
    """

    if alignment_method == "leftmost_edge" \
            or alignment_method == "huang_normalization" \
            or alignment_method == "huang_fingertip" \
            or alignment_method == "huang_leftmost":
        return alignment_method

    return ""

def compute_single_score(model, model_mask, probe, probe_mask, score_function):
    if score_function == "hamming_dist":
        return compute_hamming_dist(model, probe)[0]
    elif score_function == "always_perfect":
        return 0

def calculate_scores(idx, dataset_path, in_path=None, out_path=None, df=None):
    if in_path is not None:
        df = pd.read_csv(in_path)
    for row in df.itertuples():
        row = list(row)
        i = row[0]
        tpl = row[1:]

        ###################################################################### Useful Paths declaration
        model_path_png, probe_path_png = tuple_to_filename(tpl, idx, ".png")
        model_path, probe_path = tuple_to_filename(tpl, idx, "")

        # check if extracted feature already exists:
        fe = tpl[idx['feature_extractor']]
        alignment_method = tpl[idx['alignment']]
        fe_path = fe + '/'
        model_fe_path = dataset_path + fe_path + model_path + preprocess_alignment_method(alignment_method)
        probe_fe_path = dataset_path + fe_path + probe_path + preprocess_alignment_method(alignment_method)

        ###################################################################### Feature Extraction and caching
        # create directory if not existing
        if not os.path.isdir(dataset_path + fe):
            os.system('mkdir ' + dataset_path + fe_path)

        # load and extract features, cache them in corresponding directory
        if not isfile(model_fe_path + '.npy'):
            load_and_extract(dataset_path + model_path_png, model_fe_path, fe)

        if not isfile(probe_fe_path + '.npy'):
            load_and_extract(dataset_path + probe_path_png, probe_fe_path, fe)

        ###################################################################### Load Arrays from disk
        print("Load extracted feature model", model_path)
        model = np.load(model_fe_path + '.npy')
        model_mask = np.load(model_fe_path + '_mask.npy')
        print("Load extracted feature probe", probe_path)
        probe = np.load(probe_fe_path + '.npy')
        probe_mask = np.load(probe_fe_path + '_mask.npy')

        ###################################################################### Perform post-feature-extraction alignment
        model, model_mask, probe, probe_mask = postprocess(model, model_mask, probe, probe_mask, alignment_method)

        ###################################################################### Compute score
        score = compute_single_score(model, model_mask, probe, probe_mask, tpl[idx["score_function"]])
        print(score)
        ###################################################################### Update dataframe
        df.at[i, "score"] = score
    df.to_csv(out_path)

# assumes experiment folder with specs already exists.
def run_population_experiment(experiment_id='i', population_id='i'):
    experiment_path = experiment_dir_pref + experiment_id + "/population_" + population_id + "/"

    # load specification and idx
    f = open(experiment_path + "spec.json")
    experiment_spec = json.load(f)
    f.close()
    spec = experiment_spec["spec"]
    idx = experiment_spec["idx"]
    combination_param_pos = experiment_spec["combination_param_pos"]

    # generate empty dataset if it does not exist yet and store it
    out_path = experiment_path + "setup.csv"
    df = None
    if not isfile(out_path):
        df = dataframe_generator(spec=spec, idx=idx, combination_parameter_pos=combination_param_pos, out=out_path)

    # calculate scores of dataframe
    scores_out_path = experiment_path + "results.csv"
    dataset_path = dataset_dir_pref + spec["dataset_id"][0] + "/"
    calculate_scores(idx, dataset_path=dataset_path, in_path=out_path, out_path=scores_out_path, df=df)

run_population_experiment(experiment_id='i', population_id='i')