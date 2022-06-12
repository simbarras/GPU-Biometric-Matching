import pandas as pd

q0 = pd.read_csv("population_i/results.csv")
q1 = pd.read_csv("population_ii/results.csv")
q2 = pd.read_csv("population_iii/results.csv")
q3 = pd.read_csv("population_iv/results.csv")

q_comb = q0.copy()
avg = 0
for i in range(q_comb.shape[0]):
    a0 = q0.iloc[i]["distance"]
    a1 = q1.iloc[i]["distance"]
    a2 = q2.iloc[i]["distance"]
    a3 = q3.iloc[i]["distance"]

    print(min(a0, a1, a2, a3))
    avg += min(a0, a1, a2, a3)
    q_comb.at[i, "distance"] = min(a0, a1, a2, a3)

q_comb.to_csv("population_v/results.csv")
print(avg / q_comb.shape[0])