import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


file_path1 = "label_properties 86 n2_1.csv"
file_path2 = "label_properties 43 N2 20x z-stack tile scan.csv"


df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
print(len(df1["Area_um2"]))

x_values1 = df1["Channel4_intensity"]
y_values1 = df1["Channel2_intensity"]
size1 = df1["Area_um2"]

x_values2 = df2["Channel4_intensity"]
y_values2 = df2["Channel2_intensity"]
size2 = df2["Area_um2"]

#data = y_values/x_values
data1 = size1
len_size1 = [1] * len(size1)
data1 = data1.to_numpy()
data1 = data1.reshape(-1,1)

data2 = size2
len_size2 = [1] * len(size2)
data2 = data2.to_numpy()
data2 = data2.reshape(-1,1)



fig, ax = plt.subplots()

vp1 = ax.violinplot(data1, [1], showmeans = True, showextrema = False)
vp2 = ax.violinplot(data2, [3], showmeans = True, showextrema = False)
plt.grid(True)
plt.xticks([1,2,3], ["86 n2", "", "43 n2"])
plt.ylabel("Nucleus area ($\mu$m^2)")

plt.show()