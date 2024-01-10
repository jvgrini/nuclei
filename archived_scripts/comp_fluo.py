import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import ttest_ind


file_path1 = "label_properties P9 6 CONTRA MIX4.csv"
file_path2 = "label_properties P9 6 IPSII MIX4.csv"


df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)


channel3_ipsii = df2["Channel3_intensity"]
channel3_contra = df1["Channel3_intensity"]

t_statistic, p_value = ttest_ind(channel3_ipsii, channel3_contra)

print(f"IPSII average: {np.average(channel3_ipsii)}")
print(f"CONTRA average: {np.average(channel3_contra)}")

print("t-statistic:", t_statistic)
print("p-value:", p_value)


ig, ax = plt.subplots()

vp1 = ax.violinplot(channel3_ipsii, [1], showmeans = True, showextrema = False)
vp2 = ax.violinplot(channel3_contra, [3], showmeans = True, showextrema = False)
plt.grid(True)
plt.xticks([1,2,3], ["IPSII","","CONTRA"])
plt.ylabel("avg fluorescence")
plt.title("G4 fluorescence")
plt.show()
