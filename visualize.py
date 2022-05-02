import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import roman

colors_muted = sns.color_palette("muted")
colors_bright = sns.color_palette("pastel")

def show_histogram(populations, labels):
    sns.set(style="darkgrid")
    sns.set_palette("pastel")

    for p in populations:
        df= pd.read_csv("population_" + p + "/results.csv")
        sns.histplot(data=df, x="distance", stat="probability", color=colors_muted[(roman.fromRoman(p.upper()) - 1) % 10],
                     label=labels[roman.fromRoman(p.upper()) - 1], kde=True, fill=True,
                     common_norm=False, common_bins=False, cumulative=False, bins=30)
    plt.legend()
    plt.show()


def reduce_distr(df, n_small, n_large):
    k = n_large / n_small
    if k <= 2:
        return df.sample(n_small)

    k = math.floor(k)
    df = df[::k]
    return df.sample(n_small)


def get_eer_confusion(pop_left, pop_right, mode="distance", cam=None):
    """
    A function to compute values in confusion matrix (only tpr, fpr) and equal error rate, given two distributions.
    @param pop_left: left distribution
    @param pop_right: right distribution
    @param mode: Imposter and Client distributions are either compared by "distance" or "similarity" metrics.
    E.g. if "distance", assumes that left distribution is client distribution (positives).
    @return: (eer), (true positive rate), (false positive rate)
    """
    if type(pop_left) == str:
        df_left = pd.read_csv("population_" + pop_left + "/results.csv")
    else:
        df_left = pop_left

    if type(pop_right) == str:
        df_right = pd.read_csv("population_" + pop_right + "/results.csv")
    else:
        df_right = pop_right

    if cam is not None:
        df_left = df_left[df_left["camera_m"] == cam]
        df_right = df_right[df_right["camera_m"] == cam]

    n_left = df_left.shape[0]
    n_right = df_right.shape[0]
    n = n_left

    # if distribution not equivalent, take every k-th element from larger distribution and subsample rest of difference.
    # this roughly preserves the original distribution.
    if n_left < n_right:
        n = n_left
        df_right = reduce_distr(df_right, n_left, n_right)

    elif n_right < n_left:
        n = n_right
        df_left = reduce_distr(df_left, n_right, n_left)

    if mode == "similarity":
        df_left["distance"] *= -1
      #  df = df_left
        df_right["distance"] *= -1
      #  df_left = df_right
     #  df_right = df
     #  print(df_left)
     #  print(df_right)

    df_left.sort_values("distance", inplace=True)
    df_right.sort_values("distance", inplace=True)

    tpr = [0]
    fpr = [0]
    fnr = [1]
    true_pos = 0
    false_neg = n
    false_pos = 0
    true_neg = n
    idx_left = 0
    idx_right = 0
    while (idx_left < n and idx_right < n):
        if idx_right == n or \
                df_left.iloc[idx_left]["distance"] < df_right.iloc[idx_right]["distance"]:
            true_pos += 1
            false_neg -= 1
            idx_left += 1
        else:
            false_pos += 1
            true_neg -= 1
            idx_right += 1

        tpr.append(true_pos / (true_pos + false_neg))
        fpr.append(false_pos / (false_pos + true_neg))
        fnr.append(1 - true_pos / (true_pos + false_neg))

    # calculate equal error rate:
    idx = 0
    while fpr[idx] < fnr[idx]:
        idx += 1

    a = fpr[idx - 1]
    b = fnr[idx - 1]
    c = fpr[idx]
    d = fnr[idx]

    eer = (d - a) * ((a - b) / (c - b - d + a)) + a
    return eer, tpr, fpr

linestyles = ["solid", "dashed", "dashdot", "dotted"]
def show_roc(tpr_s, fpr_s, legends, title="ROC betw. same and different finger"):
    for i, (tpr, fpr) in enumerate(zip(tpr_s, fpr_s)):
        tpr.append(1)
        fpr.append(1)
        plt.plot(fpr, tpr, color=colors_bright[i], linestyle=linestyles[i % 4], linewidth=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(title)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    plt.legend(legends)
    plt.show()