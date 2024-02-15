import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data adapted from the provided information
# data = {
#     'Category': ['Sham', 'Sham', 'Sham', 'Sham',
#                  'Sham', 'Sham', 'HI_contralateral', 'HI_contralateral', 'HI_contralateral',
#                  'HI_contralateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral',
#                  'HI_ipsilateral', 'HI_ipsilateral'],
#     'Mouse': [6, 6, 6, 6, 7, 7, 8, 9, 9, 10, 8, 8, 9, 9, 10, 10],
#     'Slide': [15, 15, 15, 15, 18, 18, 18, 17, 17, 18, 18, 18, 17, 17, 18, 18],
#     'Non neurons': [2.331, 2.414, 2.547, 2.902, 3.104, 3.421, 3.101, 2.509, 2.499, 1.776, 3.739, 2.674, 2.603, 2.61, 1.623, 1.977],
#     'Low intensity neurons': [25.127, 23.333, 25.406, 24.361, 31.756, 30.747, 33.180, 24.020, 18.157, 16.162, 28.173, 22.343, 22.498, 20.378, 17.800, 18.146],
#     'High intensity neurons': [60.825, 61.797, 62.789, 59.801, 62.426, 62.038, 72.973, 54.876, 41.460, 38.664, 59.677, 50.635, 47.761, 40.602, 48.218, 43.720]
# }
# data = {
#     'Category': ['HI_contralateral', 'HI_contralateral', 'HI_contralateral', 'HI_contralateral', 'HI_contralateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'Sham', 'Sham', 'Sham', 'Sham', 'Sham', 'Sham'],
#     'Non Neurons': [3869, 3883, 4171, 4010, 3889, 5866, 4242, 4430, 4359, 4338, 3814, 5467, 4019, 4252, 4208, 3992, 3722],
#     'Immature Neurons': [2313, 1864, 1948, 1467, 1661, 2314, 1734, 2037, 1769, 1796, 2012, 2054, 1742, 1783, 1716, 2011, 2146],
#     'Mature Neurons': [1161, 1099, 1148, 997, 1230, 582, 377, 575, 531, 176, 256, 1295, 1122, 976, 1190, 1277, 852]
# }

# df = pd.DataFrame(data)
# df['Total Nuclei'] = df['Non Neurons'] + df['Immature Neurons'] + df['Mature Neurons']

# # Calculate proportion of non-neurons
# df['Non Neurons Proportion'] = df['Non Neurons'] / df['Total Nuclei']

# # Specify order of categories
# category_order = ['Sham', 'HI_contralateral', 'HI_ipsilateral']

# # Create boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Category', y='Non Neurons Proportion', data=df, order=category_order, palette="Set3")
# plt.xlabel('Category')
# plt.ylabel('Proportion of Non Neurons')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Create a DataFrame

# # Melt the DataFrame to long format
# df_melted = df.melt(id_vars=['Category'], var_name='Cluster', value_name='Mean Value')

# # Plot using Seaborn
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Cluster', y='Mean Value', hue='Category', data=df_melted,palette='Set3')
# plt.xlabel('Cluster')
# plt.ylabel('Median NeuN Fluorescence')
# plt.legend(title='Category')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

mean_neun_sham = [17.469833327071168, 18.880193830249087, 18.267008549029036, 18.747223534359854, 22.857530009932944, 21.24944794202986]
mean_neun_contra = [25.371495295468804, 17.96006897620603, 14.092589094683278, 12.121970275174787, 14.522632849916398]
mean_neun_ipsi = [15.572178882269686, 12.450467134844924, 13.73823592509661, 11.754948466042773, 8.917747646283967, 13.731608283534007]

# Combine data into a single DataFrame
data = {
    'Group': ['Sham'] * len(mean_neun_sham) + ['Contralateral'] * len(mean_neun_contra) + ['Ipsilateral'] * len(mean_neun_ipsi),
    'Mean Neun': mean_neun_sham + mean_neun_contra + mean_neun_ipsi
}

df = pd.DataFrame(data)

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Mean Neun', data=df, palette="Set3")
plt.xlabel("")
plt.ylabel('Mean NeuN Fluorescence')
plt.show()