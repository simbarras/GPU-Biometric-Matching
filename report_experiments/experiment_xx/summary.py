from experiment_setup import *
from visualize import *
abbreviations = {
    "miura_distance": "miura",
    "skeleton_hd": "skel",
    "random_subsampling_dist": "rsd",
    "hamming_dist": "hd",
    "fingerfocus": "fing",
    "edge": "edge",
    "id": "id",
    "huang_normalization": "huang",
    "huang_fingertip": "h + f",
    "huang_leftmost": "h + l",
    "hist_eq": "hist",
    "skeletonize": "skel",
    "center_of_mass": "CoM",
    "miura_matching": "miura",
    "translation": "trans"
}

def abbreviate(val):
    if val in abbreviations.keys():
        return abbreviations[val]
    else:
        return val


df = pd.read_csv("summary.csv")
idx = df.index.copy()
df.sort_values("eer", inplace=True)
df = df.applymap(abbreviate)
df["eer"] = df["eer"].round(3)
df = df.iloc[: , 1:]
df.to_csv("summary_sorted.csv", index=False)
df = pd.read_csv("summary_sorted.csv")
df.to_csv("summary_sorted.csv", index=True)
sns.boxplot(x=df["distance_function"], y=df["eer"])
plt.show()
sns.violinplot(x=df["distance_function"], y=df["eer"])
plt.show()
print(df)