import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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
    
def plotNeuronsRegions(imageObjects, title=""):
    
    
    data = {'Region': [], 'Class': [], 'Count': []}
    class_names = ['Non neurons', 'Immature neurons', 'Mature neurons']
    for obj in imageObjects:
        for region, clusters in zip(['CA1', 'CA3', 'DG'], [obj.ca1Clusters, obj.ca3Clusters, obj.dgClusters]):
            for i, class_nuclei in enumerate(clusters):
                class_name = class_names[i]
                class_count = len(class_nuclei)
                data['Region'].extend([region] * class_count)
                data['Class'].extend([class_name] * class_count)
                data['Count'].extend([class_count] * class_count)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Region', y='Count', hue='Class')
    #sns.stripplot(data=df, x='Region', y='Count', hue='Class', dodge=True, jitter=True, marker='o', alpha=0.5, color='black')
    plt.title(title)
    plt.xlabel('Region')
    plt.ylabel('Number of Nuclei')
    plt.grid(True)
  
    plt.legend()
    plt.show()

def plotNeuronsRegionsbyRegion(contra, ipsi, sham, title=""):
    regions = ['CA1', 'CA3', 'DG']
    clusters = ['ca1Clusters', 'ca3Clusters', 'dgClusters']
    conditions = [sham, contra, ipsi]

    for i, region in enumerate(regions):
        plt.figure()
        plt.title(f'{title} - {region}')
        plt.xlabel('Clusters')
        plt.ylabel('Number of Neurons')

        # Creating a DataFrame for Seaborn
        data = []
        for condition, label in zip(conditions, ['Sham', 'Contra', 'Ipsi']):
            for obj in condition:
                for idx, cluster in enumerate(getattr(obj, clusters[i]), start=1):
                    if idx == 1:
                        cluster_label = "non neurons"
                    elif idx == 2:
                        cluster_label = "immature neurons"
                    elif idx == 3:
                        cluster_label = "mature neurons"
                    else:
                        cluster_label = f'Cluster {idx}'
                    data.append((cluster_label, len(cluster), label))
        
        df = pd.DataFrame(data, columns=['Cluster', 'Neurons', 'Condition'])

        # Creating boxplot using Seaborn
        sns.boxplot(x='Cluster', y='Neurons', hue='Condition', data=df, palette='Set3')

        plt.legend(title='Condition')
        plt.tight_layout()
        plt.show()
def plotRegionNeuronsDensity(contra, ipsi, sham, title=""):
    regions = ['CA1', 'CA3', 'DG']
    densities_props = ['ca1Density', 'ca3Density', 'dgDensity']
    conditions = [sham, contra, ipsi]

    for i, region in enumerate(regions):
        plt.figure()
        plt.title(f'{title} - {region}')
        plt.xlabel('Clusters')
        plt.ylabel('Number of Neurons')

        # Creating a DataFrame for Seaborn
        data = []
        for condition, label in zip(conditions, ['Sham', 'Contra', 'Ipsi']):
            for obj in condition:
                for idx, density in enumerate(getattr(obj, densities_props[i]), start=1):
                    if idx == 1:
                        cluster_label = "non neurons"
                    elif idx == 2:
                        cluster_label = "immature neurons"
                    elif idx == 3:
                        cluster_label = "mature neurons"
                    else:
                        cluster_label = f'Cluster {idx}'
                    data.append((cluster_label, density, label))
        
        df = pd.DataFrame(data, columns=['Cluster', 'Neurons', 'Condition'])

        # Creating boxplot using Seaborn
        sns.boxplot(x='Cluster', y='Neurons', hue='Condition', data=df, palette='Set3')

        plt.legend(title='Condition')
        plt.tight_layout()
        plt.show()