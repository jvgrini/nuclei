import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data adapted from the provided information
data = {
    'Category': ['Sham', 'Sham', 'Sham', 'Sham',
                 'Sham', 'Sham', 'HI_contralateral', 'HI_contralateral', 'HI_contralateral',
                 'HI_contralateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral', 'HI_ipsilateral',
                 'HI_ipsilateral', 'HI_ipsilateral'],
    'Mouse': [6, 6, 6, 6, 7, 7, 8, 9, 9, 10, 8, 8, 9, 9, 10, 10],
    'Slide': [15, 15, 15, 15, 18, 18, 18, 17, 17, 18, 18, 18, 17, 17, 18, 18],
    'Non neurons': [2.331, 2.414, 2.547, 2.902, 3.104, 3.421, 3.101, 2.509, 2.499, 1.776, 3.739, 2.674, 2.603, 2.61, 1.623, 1.977],
    'Low intensity neurons': [25.127, 23.333, 25.406, 24.361, 31.756, 30.747, 33.180, 24.020, 18.157, 16.162, 28.173, 22.343, 22.498, 20.378, 17.800, 18.146],
    'High intensity neurons': [60.825, 61.797, 62.789, 59.801, 62.426, 62.038, 72.973, 54.876, 41.460, 38.664, 59.677, 50.635, 47.761, 40.602, 48.218, 43.720]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars=['Category', 'Mouse', 'Slide'], var_name='Cluster', value_name='Mean Value')

# Plot using Seaborn
plt.figure(figsize=(12, 8))
sns.boxplot(x='Cluster', y='Mean Value', hue='Category', data=df_melted,palette='Set3')
plt.xlabel('Cluster')
plt.ylabel('Median NeuN Fluorescence')
plt.legend(title='Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()