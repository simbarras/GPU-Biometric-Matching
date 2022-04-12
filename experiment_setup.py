from resources import *
import itertools
import json
import os
import random
import roman
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
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

def dataframe_generator(spec=None, idx=None, combination_parameter_pos=None, out=None):
    """ Generates a dataframe with candidate tuples. For each tuple the indicated score will be computed.
    The dataframe is then stored as a CSV in the "experiments" folder. Caches extracted features if not cached yet.
    """

    # specification defaults:
    if spec is None:
        spec = {
            "dataset_id": ['i'],                        # datasets numbered with roman numbers
            "score_function": ['hamming_dist'],         # [hamming_dist, ...]
            "feature_extractor": ['maximum_curvature_old'],     # [maximum_curvature_old, wide_line, repeated_line]
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
    df = pd.DataFrame(index=index, columns=["score"])
    if out is not None:
        df.to_csv(out)
    return df
