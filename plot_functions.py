import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def violinAndBoxplotClusters(intensityList, cluster_0_values, cluster_1_values, cluster_2_values):
    fig, ax = plt.subplots()
            # Create a violin plot for all data points
    sns.violinplot(y=np.array(intensityList).flatten(), inner=None, color="lightblue", ax=ax, zorder=1)
     # Plot boxplots for each cluster at the specified x-values
    ax.boxplot([cluster_0_values], positions=[0], widths=0.2, patch_artist=True,
            boxprops=dict(facecolor="seagreen", zorder=2, alpha=.7),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="darkgreen"),
            capprops=dict(color="darkgreen"))

    ax.boxplot([cluster_1_values], positions=[0], widths=0.2, patch_artist=True,
            boxprops=dict(facecolor="lightcoral", zorder=2, alpha=.7),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="firebrick"),
            capprops=dict(color="firebrick"))
    ax.boxplot([cluster_2_values], positions=[0], widths=0.2, patch_artist=True,
            boxprops=dict(facecolor="sandybrown", zorder=2, alpha=.7),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="tan"),
            capprops=dict(color="tan"))
    plt.xticks([0],"")
    plt.ylabel('NeuN intensity')
    plt.show()