import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats


file_path1 = "label_properties P10 6 IPSII MIX4.csv"
file_path2 = "label_properties 86 i1 crop 3D.csv"



#df1 = pd.concat([pd.read_csv(file_path1), pd.read_csv(file_path2)], ignore_index=True)
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)


x_values = df1["Channel2_intensity"]
y_values = df1["Channel3_intensity"]

res = stats.spearmanr(x_values,y_values)
print(res.statistic)
print(res.pvalue)

data_before = [x_values, y_values]
data_after = [df2["Channel2_intensity"],df2["Channel3_intensity"]]

#statistic, p_value = stats.wilcoxon(data_before, data_after)

#print(f"Wilcoxon Statistic: {statistic}")
#print(f"P-value: {p_value}")

# Compare the p-value to your chosen significance level (e.g., 0.05)
#for p in p_value:
 #   if p < 0.05:
  #      print("Reject the null hypothesis for at least one pair: There is a significant difference.")
   #     break
#else:
 #   print("Fail to reject the null hypothesis for all pairs: No significant difference observed.")
plt.scatter(x_values,y_values)
plt.show()
