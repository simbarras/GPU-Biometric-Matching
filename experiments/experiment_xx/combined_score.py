import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from visualize import *
from experiment_setup import *

def get_mode(dist_func):
    if dist_func == "skeleton_hd":
        return "similarity"
    else:
        return "distance"

params = [
    ["hamming_dist", "random_subsampling_dist", "skeleton_hd", "miura_distance"],  # distance function
    ["fingerfocus", "edge"],  # mask
    ["id", "huang_normalization", "huang_fingertip", "translation"],  # prealign
    ["id", "hist_eq"],  # preprocess
    ["id", "skeletonize"],  # postprocess
    ["id", "center_of_mass", "miura_matching"],  # postalign
]

rows = prod_index(cartesian_params=params, comb_param_pos=None)
genuine_df = pd.read_csv("population_i/results.csv")
impostor_df = pd.read_csv("population_ii/results.csv")
summary = pd.DataFrame(columns=["distance_function", "mask", "prealign", "preprocess", "postprocess", "postalign", "cam", "eer"],
                       index=range(len(rows)))

idx = 0
for distance_function, mask, prealign, preprocess, postprocess, postalign in rows:
    gen_1 = genuine_df.loc[((genuine_df["distance_function"] == distance_function)
                         & (genuine_df["mask"] == mask))
                         & (genuine_df["prealign"] == prealign)
                         & (genuine_df["preprocess"] == preprocess)
                         & (genuine_df["postprocess"] == postprocess)
                         & (genuine_df["postalign"] == postalign)
                         & (genuine_df["camera_m"] == 1)]
    gen_2 = genuine_df.loc[((genuine_df["distance_function"] == distance_function)
                         & (genuine_df["mask"] == mask))
                         & (genuine_df["prealign"] == prealign)
                         & (genuine_df["preprocess"] == preprocess)
                         & (genuine_df["postprocess"] == postprocess)
                         & (genuine_df["postalign"] == postalign)
                         & (genuine_df["camera_m"] == 2)]

    imp_1 = impostor_df.loc[((impostor_df["distance_function"] == distance_function)
                         & (impostor_df["mask"] == mask))
                         & (impostor_df["prealign"] == prealign)
                         & (impostor_df["preprocess"] == preprocess)
                         & (impostor_df["postprocess"] == postprocess)
                         & (impostor_df["postalign"] == postalign)
                         & (impostor_df["camera_m"] == 1)]
    imp_2 = impostor_df.loc[((impostor_df["distance_function"] == distance_function)
                         & (impostor_df["mask"] == mask))
                         & (impostor_df["prealign"] == prealign)
                         & (impostor_df["preprocess"] == preprocess)
                         & (impostor_df["postprocess"] == postprocess)
                         & (impostor_df["postalign"] == postalign)
                         & (impostor_df["camera_m"] == 2)]

    gen = gen_1["distance"]  + gen_2["distance"] * 2
    imp = imp_1["distance"] + imp_2["distance"] * 2
    eer, tpr, fpr = get_eer_confusion(gen, imp, get_mode(distance_function))
    #plt.suptitle(distance_function + " " + mask + " " + prealign + " " + preprocess + " " + postprocess + " " + postalign + " " + str(camera))
    #show_roc([tpr], [fpr], ["eer: " +  str(round(eer, 3))])
    summary.iloc[idx] = [distance_function, mask, prealign, preprocess, postprocess, postalign, eer]
    print(eer)
    idx += 1

print(summary)
summary.to_csv("combined_summary.csv")