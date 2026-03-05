import json
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


#training dataset
with open('papers_with_labels.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

'''
our input:

papers['H0_num'] = 0
papers['H1_num'] = 0
papers['H0_persistence_sum'] = 0.0
papers['H1_persistence_sum'] = 0.0
papers['H0_entropy'] = 0.0
papers['H1_entropy'] = 0.0
papers['H0_diag'] = []
papers['H1_diag'] = []
'''

df = pd.DataFrame(papers)

# define features
features = ['H0_num', 'H1_num', 'H0_persistence_sum', 'H1_persistence_sum',  'H1_entropy','H0_entropy']# 'H0_diag', 'H1_diag']
X = df[features].values
y = df['label'].values
import numpy as np
import pandas as pd

# check how many NaNs per column
print(df[features].isna().sum())

# check if there are any rows with NaNs
nan_rows = df[features][df[features].isna().any(axis=1)]
print("Number of rows with NaNs:", len(nan_rows))

# optionally, see the indices and values
print(nan_rows)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 0.5 means minority class will have 50% of majority
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# train a simple classifier

clf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
clf.fit(X_train_res, y_train_res)  # <-- use resampled data

# evaluate
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
'''joblib.dump(clf, 'dbsccan_topology/cl_m1.pkl')
print("Model saved to cl_m1.pkl")'''

