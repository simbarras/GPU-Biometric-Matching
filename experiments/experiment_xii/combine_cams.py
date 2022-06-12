import pandas as pd
from visualize import *

genuine = pd.read_csv("population_v/results.csv")
impostor = pd.read_csv("population_vi/results.csv")


def combine_function(a, b, tau):
    return tau * a + (1 - tau) * b
    # return min(a, b)
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


best_tau = 1
best_eer = 1

for t in range(21):
    tau = (20 - t) / 20
    gen_comb = combine_cams(genuine, tau)
    imp_comb = combine_cams(impostor, tau)
    eer, tpr, fpr = get_eer_confusion(gen_comb, imp_comb, "distance")
    if eer < best_eer:
        best_tau = tau
        best_eer = eer
        print(best_eer, best_tau)

# plt.suptitle(distance_function + " " + mask + " " + prealign + " " + preprocess + " " + postprocess + " " + postalign + " " + str(camera))
# show_roc([tpr], [fpr], ["eer: " +  str(round(eer, 3))])
print(best_eer, best_tau)
