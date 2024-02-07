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
    
def plotNeuronsRegions(ca1Clusters, ca3Clusters, dgClusters):
    
    ca1_data = [[len(cluster) for cluster in region] for region in ca1Clusters]
    ca3_data = [[len(cluster) for cluster in region] for region in ca3Clusters]
    dg_data = [[len(cluster) for cluster in region] for region in dgClusters]
    
    print(ca1_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['Neurons', 'Immature Neurons', 'Non-Neurons']
    
    for i, data in enumerate([ca1_data, ca3_data, dg_data]):
        axes[i].boxplot(data)
        axes[i].set_title(['CA1', 'CA3', 'DG'][i])
        axes[i].set_ylabel('Number of Nuclei')
        axes[i].set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()
