#!/usr/bin/python
# -*- coding=utf-8 -*-

import xgboost as xgb
import random
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
import numpy as np
import argparse
import os
import pickle


def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default='./')  # folder that stores the log
    parser.add_argument('-s', '--seed', type=int, default=27)

    args = parser.parse_args()

    random.seed(args.seed)

    param = {
        "early_stopping_rounds": 200,
        "reg_alpha": 0,
        "colsample_bytree": 1,
        "colsample_bylevel": 1,
        "scale_pos_weight": 1,
        "learning_rate": 0.3,
        "nthread": 10,
        "min_child_weight": 1,
        "n_estimators": 1000,
        "subsample": 1,
        "reg_lambda": 1,
        "seed": args.seed,
        "objective": 'binary:logistic',
        "max_depth": 6,
        "gamma": 0,
        'eval_metric': 'auc',
        'silent': 1,
        'tree_method': 'exact',
        'debug': 0,
        'use_task_gain_self': 0,
        'when_task_split': 1,
        'how_task_split': 0,
        'min_task_gain': 0.0,
        'task_gain_margin': 0.0,
        'max_neg_sample_ratio': 0.4,
        'which_task_value': 2,
        'baseline_alpha': 1.0,
        'baseline_lambda': 1.0,
        'tasks_list_': (0, 1, 2, 3, 4, 5, 6),
        'task_num_for_init_vec': 8,
        'task_num_for_OLF': 7,
    }

    folder = args.folder

    mkdir(folder)
    data_folder = './data_MT/'

    fout = open(folder+'result_baseline.csv', 'a')

    final_total_auc = np.empty(0)
    final_total_logloss = np.empty(0)
    final_total_f1 = np.empty(0)

    for event in range(47):

        # load data
        dtrain = xgb.DMatrix(data_folder + '{}_train.data'.format(event))
        dtest = xgb.DMatrix(data_folder + '{}_val.data'.format(event))
        deval = xgb.DMatrix(data_folder + '{}_val.data'.format(event))
        
        # train
        evallist = [(dtrain, 'train'), (deval, 'eval')]
        bst = xgb.train(param, dtrain, 1000 , early_stopping_rounds= 200, evals=evallist)

        y_real = dtest.get_label()
        y_score = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        
        # Predict binary outcomes instead of probabilities
        y_pred = [1 if score >= 0.5 else 0 for score in y_score]

        # compute ROC
        fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
        all_roc_auc = auc(fpr, tpr)
        all_logloss = log_loss(y_real, y_score)
        all_f1_score = f1_score(y_real, y_pred, average='macro')

        final_total_auc = np.append(final_total_auc, all_roc_auc)
        final_total_logloss = np.append(final_total_logloss, all_logloss)
        final_total_f1 = np.append(final_total_f1, all_f1_score)

        fout.write("Round{}, ROCAUC{}, LOGLOSS{}, F1 SCORE{},\n".format(event, all_roc_auc, all_logloss, all_f1_score))

    fout.write("TOTAL All round, ROCAUC{}, LOGLOSS{}, F1 SCORE{},\n".format(final_total_auc, final_total_logloss, final_total_f1))
    fout.write("Mean Auc {},  Mean logloss {}, Mean F1 {},\n".format(np.mean(final_total_auc), np.mean(final_total_logloss), np.mean(final_total_f1)))
    fout.close()