from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd
import random,time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
import sys
sys.path.append(os.path.abspath('..'))
from Measure import *
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric


def egreduction(base,df,keyword="sex",rep=10):
    sc = MinMaxScaler()
    privileged_groups = [{keyword: 1}]
    unprivileged_groups = [{keyword: 0}]
    acc, pre, recall, f1 = [], [], [], []
    aod, eod, spd, di,fr = [], [], [], [],[]

    for i in range(rep):
        start = time.time()
        dataset_orig = BinaryLabelDataset(df=df, label_names=['Probability'], protected_attribute_names=[keyword])
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7],shuffle=True, seed=i)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.34],shuffle=True, seed=i)


        # Train on simple model 
        X_train = sc.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        w_train = dataset_orig_train.instance_weights.ravel()

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
        # dataset_orig_valid_pred.features = X_valid
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
        
        
        # train on specialized model

        X_train = sc.fit_transform(dataset_orig_train.features)
        #dataset_orig_train.features = X_train
        #dataset_orig_test.features = sc.transform(dataset_orig_test.features)
        y_train = dataset_orig_train.labels.ravel()
        w_train = dataset_orig_train.instance_weights.ravel()
        estimator = copy.deepcopy(base)

        lmod = ExponentiatedGradientReduction(estimator=estimator, 
                                              constraints="EqualizedOdds",
                                              drop_prot_attr=False)
        lmod.fit(dataset_orig_train)
        y_train_pred = lmod.predict(dataset_orig_test)

        dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = sc.fit_transform(dataset_transf_test_pred.features)
        y_test = dataset_transf_test_pred.labels

        egreduction_metric = ClassificationMetric(dataset_orig_test,
                                                    y_train_pred,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

        
        acc.append(egreduction_metric.accuracy())
        pre.append(egreduction_metric.precision())
        recall.append(egreduction_metric.recall())
        f1.append(2*(egreduction_metric.precision()*egreduction_metric.recall()/(egreduction_metric.precision()+egreduction_metric.recall())))
        aod.append(egreduction_metric.average_odds_difference())
        eod.append(egreduction_metric.equal_opportunity_difference())
        spd.append(egreduction_metric.statistical_parity_difference())
        di.append(egreduction_metric.disparate_impact())
        # flip_rate = calculate_flip_proba(clf=lmod,X_test=X_test,keyword=keyword,threshold=best_class_thresh)
        fr.append(1)
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
    filenames = ['adult','compas-scores-two-years','bank','default','GermanData','h181','heart']
    keywords = {'adult':['sex','race'],
                'compas-scores-two-years':['sex','race'],
                'bank':['age'],
                'default':['sex'],
                'GermanData':['sex'],
                'h181':['race'],
                'heart':['age']
                }

    # base = LogisticRegression(max_iter=1000)
    base = RandomForestClassifier()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            df = pd.read_csv(fname+"_processed.csv")
            result1 = egreduction(base,df,keyword=keyword,rep=20)
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