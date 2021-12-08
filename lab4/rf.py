# %%
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score

# %% load df
dfAllFeatures, labels = loadcsv.loadFeatures()
# %% PCA
df = loadcsv.myPCA(dfAllFeatures, labels)
# %% 
X = df.drop(['labels'], axis=1)
Y = df['labels']
X = (X - X.min()) / (X.max() - X.min())
Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.3, random_state=42)
print(df.shape)
# %%
scoring = {
    'auc': 'roc_auc',
    'recall': 'recall'
}
def evalModel(model, X, Y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(model, X, Y, scoring=scoring, cv=cv, n_jobs=-1)
    print('Mean AUC: %.3f' % np.mean(scores['test_auc']))
    print('Mean REC: %.3f' % np.mean(scores['test_recall']))

# %%
rf = RandomForestClassifier(n_estimators=100)
# evalModel(rf, Xtr, Ytr)
# %%
def validateModel(model, Xtr, Ytr, Xval, Yval):
    model.fit(Xtr, Ytr)
    Ypre = model.predict_proba(Xval)[:, 1]
    print('AUC: %.3f' % roc_auc_score(Yval, Ypre))
    print('Recall: %.3f' % recall_score(Yval, Ypre))

# %%
