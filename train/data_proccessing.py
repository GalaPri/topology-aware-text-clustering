import pandas as pd
import json
import umap
import numpy as np 
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from tqdm import tqdm

papers = pd.read_json('full_dataset_vectorized.json', encoding='utf-8')
#struct of papers:
""""
        cluster": "metabolic_basic",
        "keyword": "Metabolic syndrome",
        "title": "Harmonizing the Metabolic Syndrome",
        "abstract": "A cluster of risk factors for cardiovascular disease and type 2 diabetes mellitus, which occur together more often than by chance alone, have become known as the metabolic syndrome. The include raised blood pressure, dyslipidemia (raised triglycerides lowered high-density lipoprotein cholesterol), fasting glucose, central obesity. Various diagnostic criteria been proposed different organizations over past decade. Most recently, these come from International Diabetes Federation American Heart Association/National Heart, Lung, Blood Institute. main difference concerns measure obesity, with this being an obligatory component in definition, lower Institute criteria, ethnic specific. present article represents outcome a meeting between several major attempt to unify criteria. It was agreed that there should not be component, but waist measurement would continue useful preliminary screening tool. Three abnormal findings out 5 qualify person single set cut points used all components except circumference, further work is required. In interim, national or regional circumference can used.",
        "authors": "K. G. M. M. Alberti, Robert H. Eckel, Scott M. Grundy, Paul Zimmet, James I. Cleeman, Karen A. Donato, Jean‐Charles Fruchart, W. P. T. James, Catherine M. Loria, Sidney C. Smith",
        "affiliations": "American Heart Association; American Heart Association; ; Baker Heart and Diabetes Institute; American Heart Association; American Heart Association; Williams & Associates; American Heart Association; American Heart Association; American Heart Association",
        "year": 2009.0,
        "journal": NaN,
        "doi": "https://doi.org/10.1161/circulationaha.109.192644",
        "citations": 14037,
        "text": "Metabolic syndrome Harmonizing the Metabolic Syndrome A cluster of risk factors for cardiovascular disease and type 2 diabetes mellitus, which occur together more often than by chance alone, have become known as the metabolic syndrome. The include raised blood pressure, dyslipidemia (raised triglycerides lowered high-density lipoprotein cholesterol), fasting glucose, central obesity. Various diagnostic criteria been proposed different organizations over past decade. Most recently, these come from International Diabetes Federation American Heart Association/National Heart, Lung, Blood Institute. main difference concerns measure obesity, with this being an obligatory component in definition, lower Institute criteria, ethnic specific. present article represents outcome a meeting between several major attempt to unify criteria. It was agreed that there should not be component, but waist measurement would continue useful preliminary screening tool. Three abnormal findings out 5 qualify person single set cut points used all components except circumference, further work is required. In interim, national or regional circumference can used. American Heart Association; American Heart Association; ; Baker Heart and Diabetes Institute; American Heart Association; American Heart Association; Williams & Associates; American Heart Association; American Heart Association; American Heart Association ",
        "vector": [...]    
"""
#reduce
d = 20
X = np.array([np.array(v) for v in papers['vector']])
reducer = umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_proj = reducer.fit_transform(X)
papers['reduced_vector'] = [v.tolist() for v in X_proj]

#prepare for storage
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
nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(vectors)
distances, indices = nbrs.kneighbors(vectors)

#iterate over each point and it's neighborhood
for i, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Computing local topology"):
    print('Iterating over row', i)
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

papers.to_json('papers_with_topology.json', orient='records')
print('saved')


