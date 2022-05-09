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

def combine_function(a, b, tau):
    # return tau * a + (1 - tau) * b
    return min(a, b)

def combine_cams(df_1, df_2, tau):
    combined_df = pd.DataFrame(columns=["distance"], index=range(df_1.shape[0]))

    for i in range(df_1.shape[0]):
        dist1 = df_1.iloc[i]["distance"]
        dist2 = df_2.iloc[i]["distance"]
        comb_dist = combine_function(dist1, dist2, tau)
        combined_df.iloc[i] = [comb_dist]
    return combined_df

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

gen_1 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "huang_normalization")
                     & (genuine_df["preprocess"] == "hist_eq")
                     & (genuine_df["postprocess"] == "skeletonize")
                     & (genuine_df["postalign"] == "id")
                     & (genuine_df["camera_m"] == 1)]
gen_2 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "translation")
                     & (genuine_df["preprocess"] == "hist_eq")
                     & (genuine_df["postprocess"] == "skeletonize")
                     & (genuine_df["postalign"] == "id")
                     & (genuine_df["camera_m"] == 2)]

imp_1 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "edge"))
                     & (impostor_df["prealign"] == "huang_normalization")
                     & (impostor_df["preprocess"] == "hist_eq")
                     & (impostor_df["postprocess"] == "skeletonize")
                     & (impostor_df["postalign"] == "id")
                     & (impostor_df["camera_m"] == 1)]
imp_2 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "edge"))
                     & (impostor_df["prealign"] == "translation")
                     & (impostor_df["preprocess"] == "hist_eq")
                     & (impostor_df["postprocess"] == "skeletonize")
                     & (impostor_df["postalign"] == "id")
                     & (impostor_df["camera_m"] == 2)]

best_tau = 1
best_eer = 1

for t in range(21):
    tau = (20 - t) / 20
    gen_comb = combine_cams(gen_1, gen_2, tau)
    imp_comb = combine_cams(imp_1, imp_2, tau)
    eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, get_mode("miura_distance"))
    if eer < best_eer:
        best_tau = tau
        best_eer = eer
# plt.suptitle(distance_function + " " + mask + " " + prealign + " " + preprocess + " " + postprocess + " " + postalign + " " + str(camera))
# show_roc([tpr], [fpr], ["eer: " +  str(round(eer, 3))])
print(best_eer, best_tau)
