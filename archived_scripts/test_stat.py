from scipy import stats
import numpy as np

contra_positive = [1004, 524, 579, 779]
contra_negative = [5897, 6466, 6056, 6133]
ipsii_positive = [579, 403, 469, 708]
ipsii_negative = [3558, 3288, 4468, 4940]


contra_ratio = [pos / (pos + neg) for pos, neg in zip(contra_positive, contra_negative)]
ipsii_ratio = [pos / (pos + neg) for pos, neg in zip(ipsii_positive, ipsii_negative)]

print(contra_ratio)
print(ipsii_ratio)

tstat, pvalue = stats.ttest_ind(contra_ratio, ipsii_ratio)

print("mean contra:", np.mean(contra_ratio))
print("mean ipsii: ", np.mean(ipsii_ratio))
print(f"tstat: {tstat}, pvalue: {pvalue}")