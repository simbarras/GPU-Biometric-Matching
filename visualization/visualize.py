import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_i = pd.read_csv("experiments/experiment_i/population_iv/results.csv")
df_ii = pd.read_csv("experiments/experiment_i/population_ii/results.csv")
df_iii = pd.read_csv("experiments/experiment_i/population_iii/results.csv")

sns.set(style="darkgrid")
# fig, axs = plt.subplots(5, 1, figsize=(7, 14))
#
sns.histplot(data=df_i, x="score", stat="density", color="skyblue", label="Same Person and Finger", kde=True) #, ax=axs[0])
sns.histplot(data=df_ii, x="score", stat="density", color="red", label="Different Person same Finger", kde=True) #, ax=axs[1])

sns.histplot(data=df_ii, x="score", stat="density", color=[0, 1, 1], label="Different Person different Finger", kde=True) #, ax=axs[2])
plt.legend()
# sns.violinplot(x=df_i["dataset_id"], y=df_i["score"])
plt.show()




# plt.hist(x, density=True, bins=30)  # density=False would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data')

# x = df['score']
# sns.displot(x, bins=40, kde=True)
