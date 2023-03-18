#| export https://github.com/fastai/nbdev
#!pip install jupyter_contrib_nbextensions


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


import logging
import time

def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Running {func.__name__} with args {args} and kwargs {kwargs}, took {elapsed_time:.4f} seconds")
        return result
    return wrapper

@log
def add(x, y):
    time.sleep(1)
    return x + y

result = add(2, 3)

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
        
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df        

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


from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING,partial
from sklearn.metrics import mean_squared_error, roc_auc_score

# Define searched space
hyper_space = {'metric':'auc','boosting':'gbdt','min_data_in_leaf': 99,'n_estimators':100,
               'objective': hp.choice('objective',['regression_l1','regression_l2','huber']), 
               'learning_rate': hp.choice('learning_rate', [0.05, .1, .3]),
               'max_depth': hp.choice('max_depth',np.arange(5, 25, dtype=int)),
               'num_leaves': hp.choice('num_leaves', np.arange(16, 1024, 8, dtype=int)),
               'max_bin':hp.choice('max_bin',np.arange(50, 500, 25, dtype=int)),
               'subsample': hp.uniform('subsample', 0.5, 1),
               'feature_fraction': hp.uniform('feature_fraction', 0.5, 1), # colsample_bytree
               'reg_alpha': hp.uniform('reg_alpha', 0, 1),
               'reg_lambda':  hp.uniform('reg_lambda', 0, 1),               
               'min_child_samples': hp.choice('min_child_samples',np.arange(10, 100, 10, dtype=int))}

target = "open_channels"
features=[i for i in train.columns if i not in target]

X_train, X_valid, y_train, y_valid = train_test_split(train[features], train[target], test_size=0.33, random_state=42)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)
del X_train, y_train
gc.collect()

def evaluate_metric(params):
    print(params)
    model = lgb.train(params, lgb_train,num_boost_round = 3000, valid_sets=[lgb_train, lgb_valid],
                           early_stopping_rounds=100, verbose_eval=1000)

    pred = model.predict(X_valid)
    score = -roc_auc_score(y_valid,pred) # Hyperopt tries to minimize error by default
    
    print(score)
    return {'loss': score,'status': STATUS_OK,'stats_running': STATUS_RUNNING}


# Trail
trials = Trials()
# Set algoritm parameters
algo = partial(tpe.suggest, n_startup_jobs=-1)
# Setting the number of evals
MAX_EVALS = 55
# Fit Tree Parzen Estimator
best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1,algo=algo, max_evals=MAX_EVALS, trials=trials)
# Print best parameters
best_params = space_eval(hyper_space, best_vals)
print(best_params)


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



### Permutation Importance

results = {}
results['base_score'] = base_auc
cat_cols = [col for col in x_train.select_dtypes("category").columns]    
# OOF AUC: 0.923873319287809
for col in tqdm(imp["feature"].tolist()):
    if col in cat_cols:
        continue
    freezed_col = x_valid[col].copy()
    x_valid[col] = np.random.permutation(x_valid[col])

    preds = estimator.predict(x_valid)
    results[col] = metrics.roc_auc_score(y_valid, preds)

    x_valid[col] = freezed_col

    print(f'column: {col} - {results[col]}')

    # bad_features = [k for k in results if results[k] > results['base_score'] + 0.005]
    
bad = pd.DataFrame()
bad["feature"] = results.keys()
bad["badness"] = results.values()
bad = bad.sort_values("badness",ascending=False)
bad


### LGBM Ranker

def train_loop_ranker(df, num_folds, useful_features, target, params, verbose_eval, early_stopping_rounds, topk):
    kfold = GroupKFold(n_splits = num_folds)
    oof_predictions = np.zeros((df.shape[0]))
    oof_scores = []
    holdout_predictions = []
    feature_importance = pd.DataFrame()
    rankers = []
    fold = 0
    for train_index, valid_index in kfold.split(df[useful_features], df[target], df['product_content_id']):
        print("### Fold", fold+1, "###")
        
        train = df.iloc[train_index].reset_index(drop=True)
        valid = df.iloc[valid_index].reset_index(drop=True)

        print("Train Shape:", train.shape, "Valid Shape:", valid.shape)

        query_train = train.groupby('product_content_id')['product_review_id'].count().tolist()
        query_valid = valid.groupby('product_content_id')['product_review_id'].count().tolist()

        from lightgbm import LGBMRanker
        ranker = LGBMRanker(**params)
        ranker.fit(train[useful_features], train[target], group = query_train,
                   eval_set=[(valid[useful_features], valid[target])], eval_group=[query_valid], eval_at=[topk],
                   verbose=verbose_eval, early_stopping_rounds=early_stopping_rounds)

        rankers.append(ranker)
        oof_pred = ranker.predict(valid[useful_features])
        oof_predictions[valid_index] = oof_pred

        oof_score = ranker.best_score_['valid_0'][f'ndcg@{topk}']
        print(f"Fold {fold+1} NDCG Score", oof_score,"\n")
        oof_scores.append(oof_score)            
            
        imp = pd.DataFrame(sorted(zip(ranker.feature_importances_,useful_features)), columns=["importance","feature"])                
        imp["fold"] = fold
        feature_importance = pd.concat([feature_importance, imp], axis=0)
        fold += 1 
        gc.collect()

    oof_score = np.mean(oof_scores)
    print(f"Out of fold NDCG", oof_score)
        
    return rankers, oof_predictions, feature_importance


data[target_col] = data.groupby('product_content_id')[base_col].rank(method='dense',ascending=True) - 1 

ranker_params= {'n_estimators':500,
                'n_jobs': 12,
                'learning_rate': 0.01,
                'num_leaves': 1024,
                'min_data_in_leaf': 300,
                'max_depth': -1,
                'max_bin': 255,
                'feature_fraction': 0.75,
                'bagging_fraction': 1,
                'bagging_freq': 0,
                'seed': 2019,
                'importance_type':'gain',
                'label_gain': np.arange(0,data[target_col].max()+1),
                'lambdarank_truncation_level':int(data[target_col].max())
              }

models, oof_predictions, feature_importance = train_loop_ranker(df=data[mask_click_model].reset_index(drop=True), num_folds=3, 
                                                                useful_features = feature_list, params = ranker_params, target = target_col, 
                                                                verbose_eval=100, early_stopping_rounds=None, topk=10)
