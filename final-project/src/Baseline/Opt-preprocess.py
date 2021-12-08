from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd
import random,time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath('..'))
from Measure import *
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from sklearn.preprocessing import StandardScaler




def optpreprocess(base,dataset_used, protected_attribute_used,keyword="sex",rep=10):
    if dataset_used == "adult":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_adult(['race'])
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
    elif dataset_used == "german":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_german(['sex'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            dataset_orig = load_preproc_data_german(['age'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.1,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }    
    elif dataset_used == "compas":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_compas(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_compas(['race'])
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

    #random seed
    np.random.seed(1)

    # Split into train, validation, and test
    
    sc = MinMaxScaler()
    acc, pre, recall, f1 = [], [], [], []
    aod, eod, spd, di,fr = [], [], [], [],[]

    for i in range(rep):
        start = time.time()
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
        # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


        
        not_trained = True
        while not_trained:
            try:
                OP = OptimPreproc(OptTools, optim_options,
                  unprivileged_groups = unprivileged_groups,
                  privileged_groups = privileged_groups)
                OP = OP.fit(dataset_orig_train)
                not_trained = False
            except:   # if the optimization fails, we try again
                not_trained = True
            
            

        # Transform training data and align features
        dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
        dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
        # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
        dataset_orig_test = dataset_transf_train.align_datasets(dataset_orig_test)
        # print(dataset_orig_test.features.shape)

        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
        # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())


        # Train on original data
        X_train = sc.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        

        lmod = copy.deepcopy(base)
        lmod.fit(X_train, y_train,
                 sample_weight=dataset_orig_train.instance_weights)
        y_train_pred = lmod.predict(X_train)

        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

        dataset_orig_train_pred = dataset_orig_train.copy()
        dataset_orig_train_pred.labels = y_train_pred

        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        X_valid = sc.transform(dataset_orig_valid_pred.features)
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = sc.transform(dataset_orig_test_pred.features)
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

        # find optimal thresh
        num_thresh = 100
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):
            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

            classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                                dataset_orig_valid_pred,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)

            ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() + classified_metric_orig_valid.true_negative_rate())
        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]

        # train on transformed data
        X_train = sc.fit_transform(dataset_transf_train.features)
        y_train = dataset_transf_train.labels.ravel()
        w_train = dataset_transf_train.instance_weights.ravel()

        lmod = copy.deepcopy(base)
        lmod.fit(X_train, y_train,
                 sample_weight=dataset_transf_train.instance_weights)
        y_train_pred = lmod.predict(X_train)

        dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = sc.fit_transform(dataset_transf_test_pred.features)
        y_test = dataset_transf_test_pred.labels
        dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

        for thresh in (class_thresh_arr):
            if thresh == best_class_thresh:
                fav_inds = dataset_transf_test_pred.scores > thresh
                dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
                dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label
        df_test = dataset_orig_test.convert_to_dataframe()[0]
        df_pred = dataset_transf_test_pred.convert_to_dataframe()[0]
        df_train = dataset_orig_train.convert_to_dataframe()[0]

        # Evaluate on transformed data

        X_test = (df_test.loc[:, df_test.columns != df_test.columns[-1]])
        X_train = (df_train.loc[:, df_train.columns != df_train.columns[-1]])
        y_test = df_test[df_test.columns[-1]]
        y_train = df_train[df_train.columns[-1]]
        y_pred = df_pred[df_pred.columns[-1]]
        cm = confusion_matrix(y_test, y_pred)
        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        aod.append(measure_final_score(df_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'aod'))
        eod.append(measure_final_score(df_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'eod'))
        spd.append(measure_final_score(df_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'SPD'))
        di.append(measure_final_score(df_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'DI'))
        flip_rate = calculate_flip_proba(clf=lmod,X_test=X_test,keyword=keyword,threshold=best_class_thresh)
        fr.append(flip_rate)
        print("Round", (i + 1), "finished.")
        print('Time',time.time()-start)
        # print(" Accuracy", np.round(acc[-1],3))
        # print(" Precision", np.round(pre[-1], 3))
        # print(" Recall", np.round(recall[-1], 3))
        # print(" F1", np.round(f1[-1], 3))
        # print(" AOD", np.round(aod[-1], 3))
        # print(" EOD", np.round(eod[-1], 3))
        # print(" SPD", np.round(spd[-1], 3))
        # print(" DI", np.round(di[-1], 3))
        # print(" FR", np.round(fr[-1], 3))
    res1 = [acc, pre, recall, f1, aod, eod, spd, di,fr]
    return res1


if __name__ == "__main__":
    filenames = ['compas']
    keywords = {
                'adult':['sex','race'],
                'german':['sex'],
                'compas':['race']
                }
   

    # base = LogisticRegression(max_iter=1000)
    base = RandomForestClassifier()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            result1 = optpreprocess(base,fname,keyword,keyword=keyword,rep=10)
            a, p, r, f, ao, eo, spd, di, fr = result1
            print("**"*50)
            print(fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))
            print("-FR", np.mean(fr))