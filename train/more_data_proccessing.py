import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

papers = pd.read_json('papers_with_topology.json', encoding='utf-8')

vectors = papers['reduced_vector'].to_list()
k = 2
nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(vectors)
distances, indices = nbrs.kneighbors(vectors)


edges_between_clusters = []
for i, neighbors in enumerate(indices):
    for nb in neighbors[1:]:  # skip self
        if papers.at[i, 'cluster'] != papers.at[nb, 'cluster']:
            edges_between_clusters.append((i, nb))   # these are true borders

# Collect all nodes that participate in inter-cluster edges
boundary_nodes = set(i for edge in edges_between_clusters for i in edge)

# Label papers: 0 = border, 1 = core
papers['label'] = papers.index.map(lambda x: 0 if x in boundary_nodes else 1)

# Save result
papers.to_json('papers_with_labels.json', orient='records')
print("saved with border labels!")
num_zero_H0 = (papers['H0_entropy'] == 'Nan').sum()
print("Number of rows with H0 persistence sum = 0:", num_zero_H0)

'''labels = []
for i, neighbors in enumerate(indices):
    neighbors = neighbors[1:]  # skip self
    cluster = papers.at[i, 'cluster']
    neighbor_clusters = papers.loc[neighbors, 'cluster']
    num_same_cluster = (neighbor_clusters == cluster).sum()

    # label as core (1) if more than half neighbors are same cluster
    label = 1 if num_same_cluster > k * 0.5 else 0
    labels.append(label)

papers['label'] = labels'''