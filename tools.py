import numpy as np
import pandas as pd
import torch

from sklearn import metrics
from sklearn.externals import joblib 
from sklearn.preprocessing import MinMaxScaler

# def vec(fit_data, transform_data, vectorizer, tag):
# def vec(kwargs):
#     fit_data = kwargs['fit_data']
#     transform_data = kwargs['transform_data']
#     vectorizer = kwargs['vectorizer']
#     tag = kwargs['tag']
#     index = kwargs['index']
    
#     vectorizer = vectorizer.fit(fit_data)

#     d = {'{0}_{1:04d}'.format(tag, v):'{0}_{1:04d}_{2}'.format(tag, v, k) for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])}
#     c = sorted(d.keys())

#     df = pd.DataFrame(
#         data=vectorizer.transform(transform_data).toarray(),
#         columns=c, 
#         index=index,
#         dtype=np.float16)
    
#     df = df.loc[:, (df != 0).any(axis=0)]
#     d = {k:v for k, v in d.items() if k in df.columns}
    
#     print(tag, df.shape)
    
#     res_dict = {
#         'tag': tag,
#         'columns': df.columns.tolist(),
#         'colmap': d,
# #         'data':scipy.sparse.csr_matrix(df.values),
#         'data':df,
#     }
    
#     return res_dict



def merge_preds(pred_csv, score_col='smishing'):
    if len(pred_csv) == 0:
        return 
    print(len(pred_csv), pred_csv)
#     df_test = pd.read_csv('input/public_test.csv', index_col=0)
    df_submit = pd.read_csv(pred_csv[0], index_col=0)
    df_submit[score_col] = -1
    df_submit = df_submit[[score_col]]

    for csv in pred_csv:
        df = pd.read_csv(csv, index_col=0)
        df[[score_col]] = MinMaxScaler().fit_transform(df[[score_col]].values)
        c = csv.split('__')[0].split('_')[-1]
#         print(csv, c)

        df_submit['smishing_{}'.format(c)] = df[score_col]
#         print(df_submit.columns)
        
    pred_cols = [c for c in df_submit.columns if 'smishing_' in c]
#     print(pred_cols)
    df_submit['std'] = df_submit[pred_cols].std(axis=1)
    df_submit['median'] = df_submit[pred_cols].median(axis=1)
    df_submit['mean'] = df_submit[pred_cols].mean(axis=1)

    return df_submit


def save_feature_importance(renamed_cols, bst, importance_type, file_name):
    impt_dict = {k:v for k, v in zip(renamed_cols, bst.feature_importance(importance_type=importance_type))}
    print(f'<{importance_type}>\n', sorted(impt_dict.items(), key=(lambda x:x[1]), reverse=True)[:10])
    joblib.dump(impt_dict, f'model/{file_name}_{importance_type}.pkl')

def eval_summary(y_true, y_score, cut_off=0.5):
    if len(y_true) == 0 or len(y_score) == 0:
        return 'zero length'
    if len(y_true) != len(y_score):
        return 'diff length'
    
    y_pred = y_score.copy()
    y_pred[y_pred > cut_off] = 1
    y_pred[y_pred <= cut_off] = 0

    eval_dict = {}
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    
    eval_dict['auc'] = metrics.auc(fpr, tpr)
    eval_dict['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    
    pre, rec, _, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1)
    eval_dict['precision'] = pre[1]
    eval_dict['recall'] = rec[1]
    
    return eval_dict

def report(train_index, train_true, train_score, valid_index, valid_true, valid_score, tag):
    train_dict = eval_summary(train_true, train_score, cut_off=0.5)
    print('\n\tReport <train> {}\n'.format(tag), train_dict)
    
    valid_dict = eval_summary(valid_true, valid_score, cut_off=0.5)
    print('\n\tReport <valid> {}\n'.format(tag), valid_dict)

    df_pred_train = pd.DataFrame(index=train_index, data=train_score, columns=['score'])
    df_pred_train['type'] = 'train'
    df_pred_train['smishing'] = train_true
    
    df_pred_valid = pd.DataFrame(index=valid_index, data=valid_score, columns=['score'])
    df_pred_valid['type'] = 'valid'
    df_pred_valid['smishing'] = valid_true
    
    print('train_auc - valid_auc:', train_dict['auc'] - valid_dict['auc'])
    
    df_pred_model = pd.concat([df_pred_train, df_pred_valid])
              
    df_pred_model[['score', 'smishing']].to_csv('submit/{}_score.csv'.format(tag), index=True)
    print('\n\tReport <model> {}\n'.format(tag), eval_summary(df_pred_model['smishing'].values, 
                                           df_pred_model['score'].values, cut_off=0.5))
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, min_epoch=3, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.curr_epoch = 0
        self.min_epoch = min_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = -1

    def __call__(self, val_loss, model):
        self.curr_epoch += 1
        score = -val_loss
#         self.improved = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping {self.curr_epoch} / {self.min_epoch} counter: {self.counter} out of {self.patience}')
            if (self.curr_epoch > self.min_epoch) & (self.counter >= self.patience):
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = self.curr_epoch
            self.save_checkpoint(val_loss, model)
#             self.improved = True
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...')
        torch.save(model.state_dict(), 'model/checkpoint.pt')
#         joblib.dump(model, 'checkpoint.model')
        self.val_loss_min = val_loss