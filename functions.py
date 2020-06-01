import sklearn
import pandas as pd
import time
import gc 
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split, GroupKFold, KFold, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import lightgbm as lgb
import datetime 
import pickle
import pymysql 
from sqlalchemy import create_engine
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))

def runQuery(query,connexion):
    cur = connexion.cursor()
    cur.execute(query)
    res = cur.fetchall()
    return pd.DataFrame(list(res),columns = [e[0] for e in cur.description])

class GoldenTimer:
    def __init__(self, show=True):
        self.start_time = time.time()
        self.show = show

    def time(self, print_str):
        duration = time.time() - self.start_time
        if self.show:
            print(print_str, duration)
        self.start_time = time.time()
        
def missing_fun(data):
    missing_df = data.dtypes.to_frame("type").reset_index()
    percent_missing = data.isnull().sum() * 100 / len(data)
    nunique = data.nunique(dropna=False).values
    missing_df["percent_missing"] = percent_missing.values
    missing_df["nunique"] = nunique
    missing_df = missing_df.loc[missing_df.percent_missing !=0]
    missing_df = missing_df.sort_values(by="percent_missing", ascending=False)
    missing_df["min"], missing_df["max"] = np.nan, np.nan
    missing_df.loc[missing_df["type"]!="object","min"] = missing_df.loc[missing_df["type"]!="object","index"].apply(lambda x:data[x].min())
    missing_df.loc[missing_df["type"]!="object","max"] = missing_df.loc[missing_df["type"]!="object","index"].apply(lambda x:data[x].max())
    missing_df["sample"] = missing_df["index"].apply(lambda x:data[x].value_counts(dropna=False).index.tolist())
    return missing_df

def plot_cm(y_true, y_pred, title):
    figsize=(5,5)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    
    
   
 def train_loop(df, num_folds, useful_features, target, params, num_boost_round):
    kfold = KFold(n_splits = num_folds, shuffle=True, random_state = 2019)
    # tscv = TimeSeriesSplit(n_splits=num_folds)
    oof_predictions = np.zeros((df.shape[0]))
    clfs = []
    feature_importance = pd.DataFrame()
    fold = 0
    for train_index, valid_index in kfold.split(df):
        # if(fold==0):
        #    exclude_index = max(train_index)
        print("### Fold", fold+1, "###")
        x_train = df[useful_features].loc[train_index].copy()
        x_valid = df[useful_features].loc[valid_index].copy()
        y_train = df[target].loc[train_index]
        y_valid = df[target].loc[valid_index]

        tr_data = lgb.Dataset(x_train, label=y_train)
        vl_data = lgb.Dataset(x_valid, label=y_valid)  

        estimator = lgb.train(params,tr_data,valid_sets = [tr_data, vl_data],num_boost_round=num_boost_round, verbose_eval = 100,early_stopping_rounds=100) 

        clfs.append(estimator)
        oof_pred = estimator.predict(x_valid)
        oof_predictions[valid_index] = oof_pred

        oof_score = metrics.roc_auc_score(y_valid,oof_pred)
        # print(f"Fold {fold+1} score", oof_score)

        imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),x_train.columns)), columns=["importance","feature"])                
        imp["fold"] = fold
        feature_importance = pd.concat([feature_importance, imp], axis=0)
        # timer.time(f"fold {fold+1} done")
        fold += 1  

    # oof_score = metrics.roc_auc_score(df[target].iloc[(exclude_index+1):],oof_predictions[(exclude_index+1):])
    oof_score = metrics.roc_auc_score(df[target],oof_predictions)
    print("OOF Score", oof_score)
    return clfs, oof_predictions, feature_importance
    
lgb_params = {'objective':'cross_entropy','boosting_type':'gbdt','metric':'auc','nthread':-1,'learning_rate':0.01,'tree_learner':'serial',
              'num_leaves': 15,'colsample_bytree': 0.7,'min_data_in_leaf': 150,'max_depth':-1,'subsample_freq':1,'subsample':0.8,'max_bin':255,'verbose':-1,'seed': 2019}   

clfs, oof_predictions, feature_importance = train_loop(df=data, num_folds=5, useful_features=use_cols_final, target = "deriv_is_sale", params=lgb_params,num_boost_round=10000)


def corr_count(corr_result):
    count_zero = corr_result[["level_0"]].append(corr_result[["level_1"]].rename({"level_1":"level_0"},axis=1))
    count_zero = count_zero.level_0.value_counts().reset_index().rename({"index":"level_0","level_0":"count_0"},axis=1)
    count_one = corr_result[["level_1"]].append(corr_result[["level_0"]].rename({"level_0":"level_1"},axis=1))
    count_one = count_one.level_1.value_counts().reset_index().rename({"index":"level_1","level_1":"count_1"},axis=1)
    corr_result = corr_result.merge(count_zero,"left","level_0").merge(count_one,"left","level_1")
    return corr_result

def corr_eliminate(corr_result,k):
    import random
    corr_cols = []
    corr_temp = corr_count(corr_result)
    while corr_temp["correlation"].max()>k:
        if corr_temp.iloc[0,3] > corr_temp.iloc[0,4]:
            corr_cols.append(corr_temp.iloc[0,0])
        elif corr_temp.iloc[0,3] < corr_temp.iloc[0,4]:
            corr_cols.append(corr_temp.iloc[0,1])
        else:
            corr_cols.append(corr_temp.iloc[0,random.randint(0,1)])
        corr_temp = corr_temp[(corr_temp.level_0.isin(corr_cols)==False)&(corr_temp.level_1.isin(corr_cols)==False)]
        corr_temp = corr_count(corr_temp)

    return corr_cols

### corr_result = pd.read_csv("output/corr_result.csv")
### corr_result = corr_result[corr_result.correlation<1]

        
corr_matrix = data[use_cols].corr().abs()
corr = corr_matrix.unstack().sort_values(kind = "quicksort")
del corr_matrix
gc.collect()
corr = pd.DataFrame(data=corr,columns = ['correlation'])
corr = corr[(corr.correlation > 0.7)&(corr.correlation < 1)].reset_index()
corr_result = corr.drop_duplicates(subset=['correlation']).sort_values("correlation", ascending = False)
del corr
gc.collect()    
