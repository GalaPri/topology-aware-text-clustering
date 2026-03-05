import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import umap
from ripser import ripser
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import joblib
from sklearn.metrics import adjusted_rand_score, v_measure_score, homogeneity_score, completeness_score


print("Init the model paraphrase-multilingual-mpnet-base-v2")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def vectorize_paper(papers:pd.DataFrame):
    papers['text'] = (
        papers['keyword'].fillna('') + ' ' +
        papers['title'].fillna('') + ' ' +
        papers['abstract'].fillna('') + ' ' +
        papers['affiliations'].fillna('') + ' ' +
        papers['journal'].fillna('')
    )
    papers['vector'] = list(model.encode(
    papers['text'].tolist(),
    show_progress_bar=True
))

    return papers

print('Reading metadata')

# ===== Choose your fighter =====

'''
testing dataset is 'full_dataset.json'
real case with obv clusters "full_dataset_distinct.json"
different sized clusters "full_dataset_different_size.json"
for more overlap full_dataset_different_size_overlap.json
for even more overlaps dirty.json
'''

FILEPATH = "article_data_extraction/dirty.json"
#read the metadata of papers
papers = pd.read_json(FILEPATH, encoding='utf-8')
print("Number of rows before deduplication:", len(papers))

# Remove duplicates based on 'doi'
papers = papers.drop_duplicates(subset=['doi'])

# Reset indices
papers = papers.reset_index(drop=True)

# Check number of rows after deduplication
print("Number of rows after deduplication:", len(papers))
#vectorize them
print('Vectorizing', FILEPATH)
papers = vectorize_paper(papers)
#lets save vectors just to be safe
embeddings_to_save = papers.copy()
embeddings_to_save['vector'] = embeddings_to_save['vector'].apply(lambda x: x.tolist())
'''with open('dbsccan_topology/test_dataset_vectorized.json', 'w', encoding='utf-8') as file:
    json.dump(embeddings_to_save.to_dict(orient='records'), file, ensure_ascii=False, indent=4)
print('Vectorized dataset saved in a temp rewritable file')'''


# ===== Dim red the vectores =====
print('Dimentional reduction')
d = 20
X = np.array([np.array(v) for v in papers['vector']])
reducer = umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_proj = reducer.fit_transform(X)
papers['reduced_vector'] = [v.tolist() for v in X_proj]
print('Dimentional reduction complete')



df = papers.copy()
for fraction in [0, 0.1, 0.2, 0.3]:  # exclude 10%, 20%, 30%
    papers = df.sample(frac=(1 - fraction), random_state=42).reset_index(drop=True)
    # feed df_subset to your clustering/boundary detection pipeline
    print(f"Testing with {len(papers)} papers ({fraction*100}% removed)")


    # ===== Compute topologies =====

    #count all the features
    papers['H0_num'] = 0
    papers['H1_num'] = 0
    papers['H0_persistence_sum'] = 0.0
    papers['H1_persistence_sum'] = 0.0
    papers['H0_entropy'] = 0.0
    papers['H1_entropy'] = 0.0
    papers['H0_diag'] = None
    papers['H1_diag'] = None

    #define neighborhoods
    vectors = papers['reduced_vector'].to_list()
    k = 200
    print()
    print(f'Evaluating local topology for neghiborhoods of {k} nearest elements')
    print()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    #iterate over each point and it's neighborhood
    for i, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Computing local topology"):
        #print('Iterating over row', i)
        neighbors = neighbors[1:]  # skip self
        local_vectors = np.array([vectors[j] for j in neighbors])

        #persistance diagramms
        diagrams = ripser(local_vectors, maxdim=1)['dgms']
        H0 = diagrams[0]  # connected components
        H1 = diagrams[1]  # loops
        papers.at[i, 'H0_diag'] = H0.flatten().tolist()
        papers.at[i, 'H1_diag'] = H1.flatten().tolist()
        #betti numbers
        papers.at[i, 'H0_num'] = len(H0)
        papers.at[i, 'H1_num'] = len(H1)

        H0 = diagrams[0].copy()
        pers_H0 = H0[:,1] - H0[:,0]
        pers_H0 = np.nan_to_num(pers_H0, nan=0.0, posinf=0.0, neginf=0.0)
        papers.at[i, 'H0_persistence_sum'] = np.sum(pers_H0)

        papers.at[i, 'H1_persistence_sum'] = np.sum(H1[:,1] - H1[:,0])

        # entropy calculation
        for H, col in [(H0, 'H0_entropy'), (H1, 'H1_entropy')]:
            if len(H) > 0:
                pers = H[:,1] - H[:,0]
                pers = np.nan_to_num(pers, nan=0.0, posinf=0.0, neginf=0.0)
                pers = pers[pers > 0]
                if pers.sum() > 0:
                    prob = pers / pers.sum()
                    entropy = -np.sum(prob * np.log(prob))
                    papers.at[i, col] = entropy
                else:
                    papers.at[i, col] = 0.0  # zero persistence → zero entropy
            else:
                papers.at[i, col] = 0.0  # no features → no entropy

    #papers.to_json('dbsccan_topology/test_data_with_topology.json', orient='records')

    topologies = papers.copy()
    cols = [ 
    'H0_num',
    'H1_num',
    'H0_persistence_sum',
    'H1_persistence_sum',
    'H0_entropy',
    'H1_entropy',
    'H0_diag',
    'H1_diag'
    ]
    topologies = topologies[cols]
    '''topologies.to_json('dbsccan_topology/topology.json', orient='records')
    print()
    print('Toplogies saved')
    print()'''


    # ===== Use our classifier =====

    #label points with classifier
    classifier = joblib.load('usage/cl_m1.pkl')
    features = ['H0_num', 'H1_num', 'H0_persistence_sum', 'H1_persistence_sum',  'H1_entropy','H0_entropy']
    X = papers[features].values
    y_pred_new = classifier.predict(X)

    # Map cluster names to integers
    cluster_mapping = {name: i for i, name in enumerate(papers['cluster'].unique())}
    papers['cluster_id'] = papers['cluster'].map(cluster_mapping)

    # Add predicted labels to the DataFrame
    papers['predicted_label'] = y_pred_new

    inside_idx = papers.index[papers['predicted_label'] == 1].to_list()
    border_idx = papers.index[papers['predicted_label'] == 0].to_list()

    # embeddings
    X = np.stack(papers['reduced_vector'].to_list())




    # ===== Recreate clusters from labels =====

    # inside points and their cluster IDs
    inside_points = X[inside_idx]
    inside_clusters = papers.loc[inside_idx, 'cluster_id'].to_numpy()

    # border points
    border_points = X[border_idx]

    k = 25  # can tune
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    distances, neighbors = nbrs.kneighbors(X)
    assignments = np.full(len(X), -1)
    assignments[inside_idx] = inside_clusters
    changed = True
    while changed:
        changed = False
        for i in border_idx:
            if assignments[i] != -1:
                continue  # already assigned
            # look at neighbors that are assigned
            neighbor_ids = assignments[neighbors[i][1:]]  # skip self
            assigned_neighbors = neighbor_ids[neighbor_ids != -1]
            if len(assigned_neighbors) > 0:
                # assign to majority cluster
                new_label = np.bincount(assigned_neighbors).argmax()
                assignments[i] = new_label
                changed = True
    papers['reconstructed_cluster'] = assignments




    # ===== Check accuracy =====
    # compare with true clusters
    accuracy = (papers['cluster_id'] == papers['reconstructed_cluster']).mean()
    print("Cluster reconstruction accuracy:", accuracy)



    cluster_counts = papers['reconstructed_cluster'].value_counts().sort_index()

    # Number of clusters (excluding -1)
    n_clusters = len(cluster_counts) - (1 if -1 in cluster_counts else 0)

    print(f"Number of clusters (excluding -1): {n_clusters}")
    print("\nNumber of elements per cluster (including -1 for borders):")
    print(cluster_counts)


    cluster_counts = papers['cluster'].value_counts()
    print("Number of points per true cluster:")
    print(cluster_counts)


    true_labels = papers['cluster']   # <-- Make sure your DataFrame has this column!
    pred_labels = papers['reconstructed_cluster']

    # External metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    homo = homogeneity_score(true_labels, pred_labels)
    comp = completeness_score(true_labels, pred_labels)

    print(f"\n🔍 Cluster Comparison:")
    print(f"Adjusted Rand Index     : {ari:.4f}")
    print(f"V-measure               : {v_measure:.4f}")
    print(f"Homogeneity             : {homo:.4f}")
    print(f"Completeness            : {comp:.4f}")
    print(f"---------------------------------------")
    
    
    # ===== More checks =====
    from collections import defaultdict

    # Create mapping from cluster label -> set of paper titles
    true_cluster_to_titles = defaultdict(set)
    pred_cluster_to_titles = defaultdict(set)

    for _, row in papers.iterrows():
        true_cluster_to_titles[row['cluster']].add(row['doi'])
        pred_cluster_to_titles[row['reconstructed_cluster']].add(row['doi'])

    # Compare each true cluster to predicted clusters
    for true_label, true_titles in true_cluster_to_titles.items():
        print(f"\nTrue cluster '{true_label}' ({len(true_titles)} papers):")
        
        # Compute overlap with all predicted clusters
        overlaps = []
        for pred_label, pred_titles in pred_cluster_to_titles.items():
            intersection = true_titles & pred_titles
            overlaps.append((pred_label, len(intersection), len(intersection)/len(true_titles)*100))
        
        # Sort by largest overlap
        overlaps.sort(key=lambda x: x[1], reverse=True)
        
        for pred_label, count, pct in overlaps[:3]:  # top 3 overlaps
            print(f"  Pred cluster {pred_label}: {count} papers ({pct:.1f}%)")


    # Compute overlap percentages
    overlap_percentages = {}
    for true_label, true_titles in true_cluster_to_titles.items():
        # Find predicted cluster with the most overlap
        max_overlap = 0
        best_pred = None
        for pred_label, pred_titles in pred_cluster_to_titles.items():
            intersection_size = len(true_titles & pred_titles)
            if intersection_size > max_overlap:
                max_overlap = intersection_size
                best_pred = pred_label
        pct = max_overlap / len(true_titles) * 100
        overlap_percentages[true_label] = (best_pred, pct)

    # Print results
    print("\n🔍 Cluster Overlap Percentages:")
    for true_label, (pred_label, pct) in overlap_percentages.items():
        print(f"True cluster '{true_label}' best matches predicted cluster {pred_label}: {pct:.1f}% overlap")