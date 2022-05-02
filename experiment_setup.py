import itertools
import json
import os
import shutil
import roman
from os import listdir
from os.path import isfile, join
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
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


def dataframe_generator(spec=None, idx=None, combination_parameter_pos=None, out=None, num_rows=None):
    """ Generates a dataframe with candidate tuples. For each tuple the indicated distance will be computed.
    The dataframe is then stored as a CSV in the "experiments" folder. Caches extracted features if not cached yet.
    """

    # specification defaults:
    if spec is None:
        spec = {
            "dataset_id": ['ii'],                        # datasets numbered with roman numbers
            "distance_function": ['hamming_dist'],         # [hamming_dist, ...]
            "mask": ['morph'],
            "prealign": ['id'],
            "preprocess": ['hist_eq'],
            "feature_extractor": ['maximum_curvature_old'],     # [maximum_curvature_old, wide_line, repeated_line]
            "postprocess": ['id'],
            "postalign": ['id'],                        # [id, cm, ...]
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
            "distance_function": 1,
            "mask": 2,
            "prealign": 3,
            "preprocess": 4,
            "feature_extractor": 5,
            "postprocess": 6,
            "postalign": 7,
            "id_m": 8,
            "id_p": 9,
            "side_m": 10,
            "side_p": 11,
            "finger_m": 12,
            "finger_p": 13,
            "trial_m": 14,
            "trial_p": 15,
            "camera_m": 16,
            "camera_p": 17
        }


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
            img_m, img_p = tuple_to_filename(tpl, idx, suffix='.png')

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
    df = pd.DataFrame(index=index, columns=["distance"])
    if num_rows is not None:
        df = df.sample(num_rows)

    if out is not None:
        df.to_csv(out)
    return df

def setup_experiment(experiment_id, num_rows=None):
    """
    Creates experiment folder for specified experiment if not existing already.
    If experiment already exists, write spec to next population number available.
    Loads specifications from "exp_specifications.json", creates copy in created population folder (for reference).
    Stores setup dataframe in population folder.
    """

    # create experiment folder if not existing
    experiment_path = experiment_dir_pref + experiment_id
    if not os.path.isdir(experiment_path):
        os.system('mkdir ' + experiment_path)

    # look for next population number
    i = 1
    while(True):
        p = num_to_roman(i)
        population_path = experiment_path + "/population_" + p
        if not os.path.isdir(population_path):
            os.system('mkdir ' + population_path)
            break
        i = i + 1

    # load specification and idx
    f = open("experiments/exp_specifications.json")
    experiment_spec = json.load(f)
    f.close()

    shutil.copyfile("experiments/exp_specifications.json", population_path + "/spec.json")

    spec = experiment_spec["spec"]
    idx = experiment_spec["idx"]
    combination_param_pos = experiment_spec["combination_param_pos"]

    dataframe_generator(spec=spec, idx=idx, combination_parameter_pos=combination_param_pos,
                        out=population_path + "/setup.csv", num_rows=num_rows)