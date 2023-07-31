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
        "reg_alpha": 0.53,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "scale_pos_weight": 1,
        "learning_rate": 0.001,
        "nthread": 10,
        "min_child_weight": 1,
        "n_estimators": 1000,
        "subsample": 0.7,
        "reg_lambda": 12,
        "seed": args.seed,
        "objective": 'binary:logistic',
        "max_depth": 7,
        "gamma": 0.39,
        'eval_metric': 'auc',
        'silent': 1,
        'tree_method': 'exact',
        'debug': 0,
        'use_task_gain_self': 0,
        'when_task_split': 1,
        'how_task_split': 0,
        'min_task_gain': 0.0,
        'task_gain_margin': 0.0,
        'max_neg_sample_ratio': 0.2,
        'which_task_value': 2,
        'baseline_alpha': 1.0,
        'baseline_lambda': 1.0,
        'tasks_list_': (0, 1, 2, 3, 4, 5, 6),
        'task_num_for_init_vec': 7,
        'task_num_for_OLF': 7,
    }

    folder = args.folder

    mkdir(folder)
    data_folder = './l_data_2/'

    fout = open(folder+'result_logo.csv', 'a')

    final_total_auc = np.empty(0)
    final_total_logloss = np.empty(0)
    final_total_f1 = np.empty(0)

    for event in range(4):

        # load data
        dtrain = xgb.DMatrix(data_folder + '{}_train.data'.format(event))
        dtest = xgb.DMatrix(data_folder + '{}_val.data'.format(event))
        deval = xgb.DMatrix(data_folder + '{}_val.data'.format(event))
        
        vals = [None] * 30
        for task in param['tasks_list_']:
            vals[task] = xgb.DMatrix(data_folder + str(event) + '_val_cluster_' + str(task) + '.data')
        
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


        # output
        fout.write('\n')
        for key in param:
            fout.write(str(key))
            fout.write(',{},'.format(param[key]))
        fout.write('\n')
        fout.write('task,auc,\n')
        for task in param['tasks_list_']:
            best_auc = 0.5
            best_logloss = 0
            y_real = vals[task].get_label()
            tree_num = 0
            for tree in range(2, bst.best_ntree_limit):
                y_score = bst.predict(vals[task], ntree_limit=tree)
                fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                logloss = log_loss(y_real, y_score)
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_logloss = logloss
                    tree_num = tree
            # acc = accuracy_score(y_real, y_score)

            print("task {} 's AUC={} logloss={} at {} tree".format(task, best_auc, best_logloss, tree_num))
            fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
        fout.write("Round{}, ROCAUC{}, LOGLOSS{}, F1 SCORE{},\n".format(event, all_roc_auc, all_logloss, all_f1_score))

    fout.write("TOTAL All round, ROCAUC{}, LOGLOSS{}, F1 SCORE{},\n".format(final_total_auc, final_total_logloss, final_total_f1))
    fout.write("Mean Auc {},  Mean logloss {}, Mean F1 {},\n".format(np.mean(final_total_auc), np.mean(final_total_logloss), np.mean(final_total_f1)))
    fout.close()