import pandas as pd
from experiment_setup import *

num_dfs = 16

dfs = []
for i in range(1, 17):
    #if i in [0, 1, 4, 5, 10, 11, 14, 15]:
    #    continue
    i_r = num_to_roman(i)
    dfs.append(pd.read_csv("population_" + i_r + "/results.csv"))

q_comb = dfs[0].copy()
avg = 0
for i in range(q_comb.shape[0]):
    avg += min(map(lambda df: df.iloc[i]["distance"], dfs))
    q_comb.at[i, "distance"] = min(map(lambda df: df.iloc[i]["distance"], dfs))

comb_pop = num_to_roman(17)
q_comb.to_csv("population_" + comb_pop + "/results.csv")
print(avg / q_comb.shape[0])