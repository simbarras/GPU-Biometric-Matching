import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import roman

def show_histogram(populations, colors, labels):
    sns.set(style="darkgrid")
    sns.set_palette("pastel")
    for p in populations:
        df= pd.read_csv("population_" + p + "/results.csv")
        sns.histplot(data=df, x="distance", stat="density", color=colors[roman.fromRoman(p.upper()) - 1], label=labels[roman.fromRoman(p.upper()) - 1], kde=True)  # , ax=axs[0])
    plt.legend()
    plt.show()

def get_eer_confusion(pop_left, pop_right, mode="distance"):
    """
    A function to compute values in confusion matrix (only tpr, fpr) and equal error rate, given two distributions.
    @param pop_left: left distribution
    @param pop_right: right distribution
    @param mode: Imposter and Client distributions are either compared by "distance" or "similarity" metrics.
    E.g. if "distance", assumes that left distribution is client distribution (positives).
    @return: (eer), (true positive rate), (false positive rate)
    """
    df_left = pd.read_csv("population_" + pop_left + "/results.csv")
    df_right = pd.read_csv("population_" + pop_right + "/results.csv")

    if mode == "similarity":
        df_left["distance"] *= -1
        df = df_left
        df_right["distance"] *= -1
        df_left = df_right
        df_right = df

    df_left.sort_values("distance", inplace=True)
    df_right.sort_values("distance", inplace=True)

    n_same = df_left.shape[0]
    n_different = df_right.shape[0]
    assert (n_same == n_different)

    tpr = [0]
    fpr = [0]
    fnr = [1]
    true_pos = 0
    false_neg = n_same
    false_pos = 0
    true_neg = n_different
    idx_left = 0
    idx_right = 0
    while (idx_left < n_same and idx_right < n_different):
        if idx_right == n_different or \
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
    return (1 - eer), tpr, fpr

def show_roc(tpr_s, fpr_s, legends):
    for tpr, fpr in zip(tpr_s, fpr_s):
        plt.plot(tpr, fpr)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("ROC betw. same and different finger")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    plt.legend(legends)
    plt.show()