import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_i = pd.read_csv("population_i/results.csv")
df_ii = pd.read_csv("population_ii/results.csv")
sns.set(style="darkgrid")
# fig, axs = plt.subplots(5, 1, figsize=(7, 14))
#
sns.histplot(data=df_i, x="distance", stat="density", color="skyblue", label="Same Finger", kde=True) #, ax=axs[0])
sns.histplot(data=df_ii, x="distance", stat="density", color="red", label="Different Finger", kde=True) #, ax=axs[1])
plt.legend()
plt.show()

