import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from visualize import *
from experiment_setup import *

def combine_function(a, b, tau):
    return tau * a + (1 - tau) * b
    #return min(a, b)
    # return max(a, b)

def combine_cams(df, tau):
    combined_df = pd.DataFrame(columns=["distance"], index=range(round(df.shape[0] / 2)))
    df = df.reset_index()  # make sure indexes pair with number of rows
    prev_row = None

    idx = 0
    for index, row in df.iterrows():
        if prev_row is None:
            prev_row = row
            continue

        a = prev_row["distance"]
        b = row["distance"]
        d = combine_function(a, b, tau)
        combined_df.iloc[idx] = [d]

        idx = idx + 1
        prev_row = None
    return combined_df

def get_mode(dist_func):
    if dist_func == "skeleton_hd":
        return "similarity"
    else:
        return "distance"

params = [
    ["skeleton_hd", "miura_distance"],  # distance function - "hamming_dist", "random_subsampling_dist", "skeleton_hd",
    ["edge"],  # mask
    ["id", "huang_normalization", "huang_fingertip", "translation"],  # prealign
    ["id", "hist_eq"],  # preprocess
    ["id", "skeletonize"],  # postprocess
    ["center_of_mass", "id"],  # postalign
]

rows = prod_index(cartesian_params=params, comb_param_pos=None)
genuine_df = pd.read_csv("population_i/results.csv")
impostor_df = pd.read_csv("population_ii/results.csv")

summary = pd.DataFrame(columns=["distance_function", "mask", "prealign", "preprocess", "postprocess", "postalign", "eer", "tau"],
                       index=range(len(rows)))


idx = 0
for distance_function, mask, prealign, preprocess, postprocess, postalign in rows:
    gen = genuine_df.loc[((genuine_df["distance_function"] == distance_function)
                         & (genuine_df["mask"] == mask))
                         & (genuine_df["prealign"] == prealign)
                         & (genuine_df["preprocess"] == preprocess)
                         & (genuine_df["postprocess"] == postprocess)
                         & (genuine_df["postalign"] == postalign)]
    imp = impostor_df.loc[((impostor_df["distance_function"] == distance_function)
                         & (impostor_df["mask"] == mask))
                         & (impostor_df["prealign"] == prealign)
                         & (impostor_df["preprocess"] == preprocess)
                         & (impostor_df["postprocess"] == postprocess)
                         & (impostor_df["postalign"] == postalign)]
    best_tau = 1
    best_eer = 1

    for t in range(21):
        tau = (20 - t) / 20
        gen_comb = combine_cams(gen, tau)
        imp_comb = combine_cams(imp, tau)
        eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, get_mode(distance_function))
        if eer < best_eer:
            best_tau = tau
            best_eer = eer
    #plt.suptitle(distance_function + " " + mask + " " + prealign + " " + preprocess + " " + postprocess + " " + postalign + " " + str(camera))
    #show_roc([tpr], [fpr], ["eer: " +  str(round(eer, 3))])
    summary.iloc[idx] = [distance_function, mask, prealign, preprocess, postprocess, postalign, best_eer, best_tau]
    print(distance_function, mask, prealign, preprocess, postprocess, postalign, best_eer, best_tau)
    idx += 1

print(summary)
summary.to_csv("summary_combined.csv")