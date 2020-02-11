import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader import data_loader_v2 # 자체적으로 만든 data loader version 2.0 ([데이콘 15회 대회] 데이터 설명 및 데이터 불러오기 영상 참조)

from sklearn.ensemble import RandomForestClassifier
import joblib # 모델을 저장하고 불러오는 역
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook

from sklearn.metrics import log_loss

from tools import eval_summary, save_feature_importance, merge_preds, report

train_folder = 'data/train/'
test_folder = 'data/test/'
train_label_path = 'data/train_label.csv'

train_list = os.listdir(train_folder)
test_list = os.listdir(test_folder)
train_label = pd.read_csv(train_label_path, index_col=0)

num_class = len(train_label['label'].unique())

# 모든 csv 파일의 상태_B로 변화는 시점이 같다라고 가정
# 하지만, 개별 csv파일의 상태_B로 변화는 시점은 상이할 수 있음
def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()-2) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df


event_time = 10
nrows = 100
train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, 
                           event_time=event_time, nrows=nrows)
print(train.shape)
joblib.dump(train, 'data/df_train_{}_{}.pkl'.format(event_time, nrows))

# train = joblib.load('data/df_train_10_60.pkl').reset_index()

# event_time = 10
# nrows = None
# test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=event_time, nrows=nrows)
# print(test.shape)
# joblib.dump(train, 'data/df_test_{}_{}.pkl'.format(event_time, nrows))
test = joblib.load('data/df_test_10.pkl')

fea_cols = [c for c in train.columns if c[0] == 'V']
print(len(fea_cols))

zero_cols = joblib.load('zero_cols.bin')
fea_cols = [c for c in fea_cols if c not in zero_cols]

print(len(fea_cols))

print(train['label'].value_counts(dropna=False))

model_ts = datetime.now().strftime('%Y%m%dT%H%M%S')
print('model_ts', model_ts)


params = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.01,  # the training step for each iteration
    'silent':1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'eval_metric': 'mlogloss',
    'num_class': 198,
#     'gpu_id': 0,
#     'tree_method': 'gpu_hist',
    'nthread': 64,
    'colsample_bytree':0.5,
    'colsample_bylevel':0.5,
    'colsample_bynode':0.5,
    'max_leaves': 15,
    
}
print(params)

num_round = 10000
print('num_round:', num_round)


submit_csv = []
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, random_state=81511991154 % 2**32-1, shuffle=True)

cv = 0
for train_index, valid_index in tqdm_notebook(skf.split(train.index, train['label'].values), total=n_splits, desc = 'CV'):
    
    X_train, X_test = train.loc[train_index, fea_cols], train.loc[valid_index, fea_cols] 
    y_train, y_test = train.loc[train_index,'label'], train.loc[valid_index, 'label']    
    
    print(X_train.shape, X_test.shape)
#     print(y_train.value_counts(dropna=False))
#     print(y_test.value_counts(dropna=False))

    train_set = xgb.DMatrix(X_train, label=y_train)
    val_set = xgb.DMatrix(X_test, label=y_test, )

    evals_result = {}
    
    watchlist = [(train_set, 'training'), (val_set, 'valid_1')]

    model = xgb.train(params, train_set, num_boost_round=num_round,
                    early_stopping_rounds=100, evals=watchlist, 
                      evals_result=evals_result, verbose_eval=50,
#                       xgb_model=model
                     )


    model_tag ='{}xgb_{}_{}_{}'.format(model_ts, cv, 
                                 evals_result['valid_1']['mlogloss'][model.best_iteration-1],
                                 evals_result['training']['mlogloss'][model.best_iteration-1]
                                )
    print(model_tag)

    joblib.dump(model, 'model/{}.model'.format(model_tag))

    test_set = xgb.DMatrix(test[fea_cols])
    pred = model.predict(test_set)

    submission = pd.DataFrame(data=pred)
    submission.index = test.index
    submission.index.name = 'id'
    submission = submission.sort_index()
    submission = submission.groupby('id').mean()

    csv_path = 'submit/{}.csv'.format(model_tag)
    submit_csv.append(csv_path)
    submission.to_csv(csv_path, index=True) 

    print(submission.sum(axis=1))
    print(submission)
    cv += 1
    #     break