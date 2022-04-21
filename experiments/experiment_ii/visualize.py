import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import roman

colors = ["skyblue", "maroon", "orange", "green", "grey"]
labels = ["Same Finger", "Same Finger", "Different Finger", "Different Finger"]

def show_histogram(populations):
    sns.set(style="darkgrid")
    sns.set_palette("pastel")
    for p in populations:
        df= pd.read_csv("population_" + p + "/results.csv")
        sns.histplot(data=df, x="distance", stat="density", color=colors[roman.fromRoman(p.upper()) - 1], label=labels[roman.fromRoman(p.upper()) - 1], kde=True)  # , ax=axs[0])
    plt.legend()
    plt.show()

def get_eer(pop_same, pop_different, mode="distance", show_roc=False):
    """
    Assumes df_same has as many rows as df_different (to do so, see sampling functionality in experiment setup).
    TODO: If this is not the case, samples from dataframe with larger amount of rows.
    @param df_same:
    @param df_different:
    @param mode: Either distance or similarity score
    @param show_roc:
    @return:
    """

    # assume left is positive, right is negative. Acceptance and rejection depend on mode.
    df_left = pd.read_csv("population_" + pop_same + "/results.csv")
    df_right = pd.read_csv("population_" + pop_different + "/results.csv")

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
    assert(n_same == n_different)

    tpr = [0]
    fpr = [0]
    fnr = [1]
    true_pos = 0
    false_neg = n_same
    false_pos = 0
    true_neg = n_different
    idx_left = 0
    idx_right = 0
    while(idx_left < n_same and idx_right < n_different):
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

    plt.plot(fnr)
    plt.plot(fpr)
    plt.show()

    if show_roc:
        plt.plot(tpr, fpr)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("ROC betw. same and different finger")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    # calculate equal error rate:

    idx = 0
    while fpr[idx] < fnr[idx]:
        idx += 1

    a = fpr[idx - 1]
    b = fnr[idx - 1]
    c = fpr[idx]
    d = fnr[idx]

    eer = (d - a) * ((a - b) / (c - b - d + a)) + a
    return 1 - eer

show_histogram(["ii", "iv"])
print(get_eer("ii", "iv", mode="similarity", show_roc=True))