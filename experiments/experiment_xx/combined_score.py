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


genuine_df = pd.read_csv("population_i/results.csv")
impostor_df = pd.read_csv("population_ii/results.csv")

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
best_tpr = []
best_fpr = []
gen = None
imp = None
for t in range(21):
    tau = (20 - t) / 20
    gen_comb = combine_cams(gen_1, gen_2, tau)
    imp_comb = combine_cams(imp_1, imp_2, tau)
    eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, get_mode("miura_distance"))
    if eer < best_eer:
        best_tau = tau
        best_eer = eer
        best_tpr = tpr
        best_fpr = fpr
        imp = imp_comb
        gen = gen_comb



gen_1 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "translation")
                     & (genuine_df["preprocess"] == "id")
                     & (genuine_df["postprocess"] == "id")
                     & (genuine_df["postalign"] == "miura_matching")
                     & (genuine_df["camera_m"] == 1)]
gen_2 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "translation")
                     & (genuine_df["preprocess"] == "id")
                     & (genuine_df["postprocess"] == "id")
                     & (genuine_df["postalign"] == "miura_matching")
                     & (genuine_df["camera_m"] == 2)]

imp_1 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "edge"))
                     & (impostor_df["prealign"] == "translation")
                     & (impostor_df["preprocess"] == "id")
                     & (impostor_df["postprocess"] == "id")
                     & (impostor_df["postalign"] == "miura_matching")
                     & (impostor_df["camera_m"] == 1)]
imp_2 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "edge"))
                     & (impostor_df["prealign"] == "translation")
                     & (impostor_df["preprocess"] == "id")
                     & (impostor_df["postprocess"] == "id")
                     & (impostor_df["postalign"] == "miura_matching")
                     & (impostor_df["camera_m"] == 2)]

best_tau_m = 1
best_eer_m = 1
best_tpr_m = []
best_fpr_m = []
gen_m = None
imp_m = None
for t in range(21):
    tau = (20 - t) / 20
    gen_comb = combine_cams(gen_1, gen_2, tau)
    imp_comb = combine_cams(imp_1, imp_2, tau)
    eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, get_mode("miura_distance"))
    if eer < best_eer:
        best_tau_m = tau
        best_eer_m = eer
        best_tpr_m = tpr
        best_fpr_m = fpr
        imp_m = imp_comb
        gen_m = gen_comb


gen_1 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "id")
                     & (genuine_df["preprocess"] == "id")
                     & (genuine_df["postprocess"] == "id")
                     & (genuine_df["postalign"] == "id")
                     & (genuine_df["camera_m"] == 1)]
gen_2 = genuine_df.loc[((genuine_df["distance_function"] == "miura_distance")
                     & (genuine_df["mask"] == "edge"))
                     & (genuine_df["prealign"] == "id")
                     & (genuine_df["preprocess"] == "id")
                     & (genuine_df["postprocess"] == "id")
                     & (genuine_df["postalign"] == "id")
                     & (genuine_df["camera_m"] == 2)]

imp_1 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "edge"))
                     & (impostor_df["prealign"] == "id")
                     & (impostor_df["preprocess"] == "id")
                     & (impostor_df["postprocess"] == "id")
                     & (impostor_df["postalign"] == "id")
                     & (impostor_df["camera_m"] == 1)]
imp_2 = impostor_df.loc[((impostor_df["distance_function"] == "miura_distance")
                     & (impostor_df["mask"] == "id"))
                     & (impostor_df["prealign"] == "id")
                     & (impostor_df["preprocess"] == "id")
                     & (impostor_df["postprocess"] == "id")
                     & (impostor_df["postalign"] == "id")
                     & (impostor_df["camera_m"] == 2)]

best_tau_i = 1
best_eer_i = 1
best_tpr_i = []
best_fpr_i = []
gen_i = None
imp_i = None
for t in range(21):
    tau = (20 - t) / 20
    gen_comb = combine_cams(gen_1, gen_2, tau)
    imp_comb = combine_cams(imp_1, imp_2, tau)
    eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, get_mode("miura_distance"))
    if eer < best_eer:
        best_tau_i = tau
        best_eer_i = eer
        best_tpr_i = tpr
        best_fpr_i = fpr
        imp_i = imp_comb
        gen_i = gen_comb



plt.suptitle("Best alignment methods (without Miura)")
show_roc([best_tpr, best_tpr_m, best_tpr_i], [best_fpr, best_fpr_m, best_fpr_i], ["No Miura - eer: " +  str(round(best_eer, 3)), "Miura - eer: " +  str(round(best_eer_m, 3)), "No proc. - eer: " +  str(round(best_eer, 3))])
show_histogram_df([gen, imp], ["Genuine - No Miura", "Imposter - No Miura"])
show_histogram_df([gen_m, imp_m], ["Genuine - Miura", "Imposter - Miura"])
