import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


file_path1 = "label_properties 43 n2 sum.csv"
file_path2 = "label_properties 86 i2_3.csv"


#df1 = pd.concat([pd.read_csv(file_path1), pd.read_csv(file_path2)], ignore_index=True)
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

x_values1 = df1["Channel4_intensity"]
y_values1 = df1["Channel1_intensity"]

x_values2 = df1["Channel4_intensity"]
y_values2 = df1["Channel2_intensity"]

x_values3 = df1["Channel4_intensity"]
y_values3 = df1["Channel3_intensity"]

#data = y_values/x_values
data1 = y_values1
data1 = data1.to_numpy()
data1 = data1.reshape(-1,1)

print(np.average(data1))

data2 = y_values2
data2 = data2.to_numpy()
data2 = data2.reshape(-1,1)

print(np.average(data2))

data3 = y_values3

data4 = df2["Channel4_intensity"]


fig, ax = plt.subplots()

vp1 = ax.violinplot(data1, [1], showmeans = True, showextrema = False)
vp2 = ax.violinplot(data2, [3], showmeans = True, showextrema = False)
vp3 = ax.violinplot(data2, [5], showmeans = True, showextrema = False)
plt.grid(True)
plt.xticks([1,2,3,4,5], ["EGFR VIII","","GFAP","","SOX2"])
plt.ylabel("avg fluorescence")
plt.title("86 n1")
plt.show()