from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import json
import umap
from sklearn.metrics import adjusted_rand_score, v_measure_score, homogeneity_score, completeness_score


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

#extract papers
papers = pd.read_json('article_data_extraction/full_dataset.json', encoding='utf-8') 

papers = vectorize_paper(papers)

#cause it wont json otherwise
embeddings_to_save = papers.copy()
embeddings_to_save['vector'] = embeddings_to_save['vector'].apply(lambda x: x.tolist())

with open('full_dataset_vectorized.json', 'w', encoding='utf-8') as file:
    json.dump(embeddings_to_save.to_dict(orient='records'), file, ensure_ascii=False, indent=4)




FILEPATH = 'article_data_extraction/full_dataset_different_size.json'
'''
testing dataset is 'full_dataset.json'
real case with obv clusters "full_dataset_distinct.json"
different sized clusters "full_dataset_different_size.json"
for more overlap full_dataset_different_size_overlap.json
'''
papers = pd.read_json(FILEPATH, encoding='utf-8')
papers = vectorize_paper(papers)

#again, i assume 20 is all good 
d = 20
X = np.array([np.array(v) for v in papers['vector']])
reducer = umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_proj = reducer.fit_transform(X)


print("Loaded embeddings shape:", X.shape)
print(papers)

print()
print(f"Clusteting for projection with dim = {d}")

#X_scaled = StandardScaler().fit_transform(X_proj)
X_scaled = normalize(X_proj, norm='l2')

# eps = neighborhood size, min_samples = min points per cluster
eps = [0.1, 0.15, 0.2, 0.25, 0.3]
s = [20,15,10,5]
for e in eps:
    print()
    for sample in s:
        print()
        print(f'Dbscan for eps={e} and sample={sample}')
        dbscan = DBSCAN(eps=e, min_samples=sample, metric='cosine')  
        labels = dbscan.fit_predict(X_scaled)

        papers['dbscan_cluster'] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"DBSCAN found {n_clusters} clusters, with {n_noise} noise points")
        print(papers[['title','dbscan_cluster']].head(10))
        
        # Compare DBSCAN results with your true labels
        true_labels = papers['cluster']   # <-- Make sure your DataFrame has this column!
        pred_labels = papers['dbscan_cluster']

        # External metrics
        ari = adjusted_rand_score(true_labels, pred_labels)
        v_measure = v_measure_score(true_labels, pred_labels)
        homo = homogeneity_score(true_labels, pred_labels)
        comp = completeness_score(true_labels, pred_labels)

        print(f"\n🔍 Cluster Comparison for eps={e}, min_samples={sample}:")
        print(f"Adjusted Rand Index     : {ari:.4f}")
        print(f"V-measure               : {v_measure:.4f}")
        print(f"Homogeneity             : {homo:.4f}")
        print(f"Completeness            : {comp:.4f}")
        print(f"---------------------------------------")
