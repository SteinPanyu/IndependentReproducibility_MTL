import xgboost as xgb
import random
from sklearn.metrics import roc_curve, auc, log_loss, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import argparse
import os
import pickle
import gc


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
    folder = args.folder
    mkdir(folder)
    data_folder = './data_MT/'

    # Hyperparameter grid
    max_depths = [5, 6, 7, 8, 9, 10, 11, 13]
    learning_rates = [0.01, 0.1, 0.3, 0.001, 0.5, 0.05]
    n_estimators_values = [200]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytrees = [0.5, 0.6, 0.8, 0.9, 1.0]
    min_child_weights = [1, 2, 3, 4, 5, 6, 7]
    colsample_bylevels = [0.5, 0.6, 0.8, 0.9, 1.0]
    gammas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45]
    reg_alphas = [0.00001, 0.0005, 0.001, 0.1, 1]
    reg_lambdas = [0.001, 0.01, 0.1, 1, 10, 12, 100]
    max_neg_sample_ratios = [0.2, 0.4, 0.5]
    early_stopping_rounds_values = [80]

    total_iterations = len(max_depths) * len(learning_rates) * len(n_estimators_values) * len(subsamples) * len(colsample_bytrees) * len(min_child_weights) * len(colsample_bylevels) * len(gammas) * len(reg_alphas) * len(reg_lambdas) * len(max_neg_sample_ratios) * len(early_stopping_rounds_values)

    best_roc_auc = 0  # Assuming higher is better
    best_params = None

    fout = open(folder+'param_test.csv', 'a')

    fout.write("Itterations: {}\n".format(total_iterations))
    dtrain = xgb.DMatrix(data_folder + '0_train.data')
    dtest = xgb.DMatrix(data_folder + '0_val.data')
    deval = xgb.DMatrix(data_folder + '0_val.data')

    # Grid search loop
    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_values:
                for subsample in subsamples:
                    for colsample_bytree in colsample_bytrees:
                        for min_child_weight in min_child_weights:
                            for colsample_bylevel in colsample_bylevels:
                                for gamma in gammas:
                                    for reg_alpha in reg_alphas:
                                        for reg_lambda in reg_lambdas:
                                            for max_neg_sample_ratio in max_neg_sample_ratios:
                                                for early_stopping_rounds in early_stopping_rounds_values:

                                                    param = {
                                                        "early_stopping_rounds": early_stopping_rounds,
                                                        "reg_alpha": reg_alpha,
                                                        "colsample_bytree": colsample_bytree,
                                                        "colsample_bylevel": colsample_bylevel,
                                                        "scale_pos_weight": 1,
                                                        "learning_rate": learning_rate,
                                                        "nthread": -1,
                                                        "min_child_weight": min_child_weight,
                                                        "n_estimators": n_estimators,
                                                        "subsample": subsample,
                                                        "reg_lambda": reg_lambda,
                                                        "seed": args.seed,
                                                        "objective": 'binary:logistic',
                                                        "max_depth": max_depth,
                                                        "gamma": gamma,
                                                        'eval_metric': 'auc',
                                                        'silent': 1,
                                                        'tree_method': 'exact',
                                                        'debug': 0,
                                                        'use_task_gain_self': 0,
                                                        'when_task_split': 1,
                                                        'how_task_split': 0,
                                                        'min_task_gain': 0.0,
                                                        'task_gain_margin': 0.0,
                                                        'max_neg_sample_ratio': max_neg_sample_ratio,
                                                        'which_task_value': 2,
                                                        'baseline_alpha': 1.0,
                                                        'baseline_lambda': 1.0,
                                                        'tasks_list_': (0, 1, 2),
                                                        'task_num_for_init_vec': 3,
                                                        'task_num_for_OLF': 2,
                                                    }

                                                    # Train
                                                    evallist = [(dtrain, 'train'), (deval, 'eval')]
                                                    bst = xgb.train(param, dtrain, n_estimators, early_stopping_rounds=early_stopping_rounds, evals=evallist)

                                                    y_score = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
                                                    y_real = dtest.get_label()
                                                    y_pred = [1 if score >= 0.5 else 0 for score in y_score]

                                                    # Compute metrics
                                                    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
                                                    all_roc_auc_current = auc(fpr, tpr)

                                                    # Log the results and hyperparameters
                                                    fout.write(" ROCAUC:{}, max_depth:{}, learning_rate:{}, n_estimators:{}, subsample:{}, colsample_bytree:{}, min_child_weight:{}, colsample_bylevel:{}, gamma:{}, reg_alpha:{}, reg_lambda:{}, max_neg_sample_ratio:{}, early_stopping_rounds:{}\n".format(all_roc_auc_current, max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight, colsample_bylevel, gamma, reg_alpha, reg_lambda, max_neg_sample_ratio ,early_stopping_rounds))
                                                    # Check if current ROC AUC is better
                                                    if all_roc_auc_current > best_roc_auc:
                                                        best_roc_auc = all_roc_auc_current
                                                        best_params = {
                                                            "early_stopping_rounds": early_stopping_rounds,
                                                            "reg_alpha": reg_alpha,
                                                            "colsample_bytree": colsample_bytree,
                                                            "colsample_bylevel": colsample_bylevel,
                                                            "scale_pos_weight": 1,
                                                            "learning_rate": learning_rate,
                                                            "nthread": 10,
                                                            "min_child_weight": min_child_weight,
                                                            "n_estimators": n_estimators,
                                                            "subsample": subsample,
                                                            "reg_lambda": reg_lambda,
                                                            "seed": args.seed,
                                                            "objective": 'binary:logistic',
                                                            "max_depth": max_depth,
                                                            "gamma": gamma,
                                                            'eval_metric': 'auc',
                                                            'silent': 1,
                                                            'tree_method': 'exact',
                                                            'debug': 0,
                                                            'use_task_gain_self': 0,
                                                            'when_task_split': 1,
                                                            'how_task_split': 0,
                                                            'min_task_gain': 0.0,
                                                            'task_gain_margin': 0.0,
                                                            'max_neg_sample_ratio': max_neg_sample_ratio,
                                                            'which_task_value': 2,
                                                            'baseline_alpha': 1.0,
                                                            'baseline_lambda': 1.0,
                                                            'tasks_list_': (0, 1, 2),
                                                            'task_num_for_init_vec': 3,
                                                            'task_num_for_OLF': 2,
                                                        }
                                        fout.write("GARBAGE COLLECTED\n")
                                        gc.collect()                                                    

    fout.close()

    # Print best hyperparameters
    print("Best ROC AUC: ", best_roc_auc)
    print("Best Hyperparameters: ", best_params)
