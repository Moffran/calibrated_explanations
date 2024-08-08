# pylint: disable=invalid-name, line-too-long, duplicate-code, cell-var-from-loop, consider-using-enumerate, missing-function-docstring
"""
Experiment used in the introductory paper to evaluate the stability and robustness of the explanations
"""

import time
import warnings
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from venn_abers import VennAbers
from shap import Explainer # version 0.44.0
from lime.lime_tabular import LimeTabularExplainer # version 0.2.0.1
from calibrated_explanations import CalibratedExplainer



# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------

test_size = 20 # number of test samples per dataset
is_debug = True
num_rep = 30

descriptors = ['uncal','va',]#,'va'
Descriptors = {'uncal':'Uncal','va': 'VA'}
models = ['xGB','RF'] # ['xGB','RF','DT','SVM',] # 'NN',

# pylint: disable=line-too-long
datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1"}
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]#
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {}
results['num_rep'] = num_rep
results['test_size'] = test_size
for dataset in klara:
    dataSet = datasets[dataset]

    tic_data = time.time()
    print(dataSet)
    fileName = 'data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=';')
    Xn, y = df.drop('Y',axis=1), df['Y']

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15) # Changed from min_leaf=4
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True)
    s2 = SVC(probability=True)
    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    h1 = HistGradientBoostingClassifier()
    h2 = HistGradientBoostingClassifier()
    g1 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='logloss')
    g2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='logloss')

    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn),'SVM': (s1,s2,"SVM",Xn),'DT': (t1,t2,"DT",Xn),'HGB': (h1,h2,"HGB",Xn)}#,'NN': (a1,a2,"NN",Xn)
    model_struct = [model_dict[model] for model in models]
    results[dataSet] = {}
    for c1, c2, alg, X in model_struct:
        tic_algorithm = time.time()
        debug_print(dataSet+' '+alg)
        results[dataSet][alg] = {}

        calibrators = {}
        for desc in descriptors:
            calibrators[desc] = {}
            calibrators[desc]['ce'] = []
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size,random_state=42)
        X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=42)

        c2.fit(X_prop_train,y_prop_train)

        calibrators['uncal']['model'] = c2
        if 'va' in descriptors:
            calibrators['va']['model'] = VennAbers()
            calibrators['va']['model'].fit(c2.predict_proba(X_cal), y_cal)
        calibrators['data'] = {'X_prop_train':X_prop_train,'y_prop_train':y_prop_train,'X_cal':X_cal,'y_cal':y_cal,'X_test':X_test,'y_test':y_test,}

        np.random.seed(1337)
        categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]

        ce = CalibratedExplainer(calibrators['uncal']['model'], X_cal, y_cal, \
            feature_names=df.columns, categorical_features=categorical_features)

        stability =  {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        stab_timer = {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        robustness = {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[], 'proba':[], 'proba_va':[]}
        rob_timer =  {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        i = 0
        while i < num_rep:
            try:
                # print(f'{i}:',end='\t')

                ce.set_random_state(i)
                tic = time.time()
                explanations = ce.explain_counterfactual(X_test)
                ct = time.time()-tic
                stab_timer['ce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                stability['ce'].append([f.feature_weights for f in explanations])

                ce.set_random_state(i)
                tic = time.time()
                explanations = ce.explain_factual(X_test)
                ct = time.time()-tic
                stab_timer['cce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                stability['cce'].append([f.feature_weights for f in explanations])

                lime = LimeTabularExplainer(X_cal, feature_names=df.columns, random_state=i)
                model = calibrators['uncal']['model']
                tic = time.time()
                lime_exps_cl = []
                for instance in X_test:
                    exp = lime.explain_instance(instance, model.predict_proba)
                    lime_exps_cl.append(exp)
                ct = time.time()-tic
                stab_timer['lime'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                tmps = []
                for j in range(len(lime_exps_cl)):
                    tmp = np.zeros(no_of_features)
                    for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                        tmp[f[0]] = f[1]
                    tmps.append(tmp)
                stability['lime'].append(tmps)

                model = calibrators['va']['model']
                tic = time.time()
                lime_exps_cl = []
                for instance in X_test:
                    exp = lime.explain_instance(instance, lambda x: model.predict_proba(x)[0])
                    lime_exps_cl.append(exp)
                ct = time.time()-tic
                stab_timer['lime_va'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                tmps = []
                for j in range(len(lime_exps_cl)):
                    tmp = np.zeros(no_of_features)
                    for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                        tmp[f[0]] = f[1]
                    tmps.append(tmp)
                stability['lime_va'].append(tmps)

                shap = Explainer(lambda x: calibrators['uncal']['model'].predict_proba(x)[:,1], X_cal, \
                    feature_names=df.columns)
                shap.random_state = i
                tic = time.time()
                explanations = shap(X_test)
                ct = time.time()-tic
                stab_timer['shap'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                stability['shap'].append(explanations.values) # pylint: disable=no-member

                shap_va = Explainer(lambda x: calibrators['va']['model'].predict_proba(x)[0][:,1], X_cal, \
                    feature_names=df.columns)
                shap.random_state = i
                tic = time.time()
                explanations = shap_va(X_test)
                ct = time.time()-tic
                stab_timer['shap_va'].append(ct)
                # print(f'{ct:.1f}')
                stability['shap_va'].append(explanations.values) # pylint: disable=no-member

                i = i + 1
            except Exception as e: # pylint: disable=broad-exception-caught
                warnings.warn(f'Error: {e}')
            print('_',end='')
        print('')
        results[dataSet][alg]['stability'] = stability
        results[dataSet][alg]['stab_timer'] = stab_timer

        i = 0
        while i < num_rep:
            np.random.seed(i)
            if alg == 'xGB':
                c2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='logloss', random_state=i)
            else:
                c2 = RandomForestClassifier(n_estimators=100, random_state=i)
            X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=i)

            c2.fit(X_prop_train,y_prop_train)
            calibrators['uncal']['model'] = c2
            if 'va' in descriptors:
                calibrators['va']['model'] = VennAbers()
                calibrators['va']['model'].fit(c2.predict_proba(X_cal), y_cal)
            ce = CalibratedExplainer(calibrators['uncal']['model'], X_cal, y_cal, \
                feature_names=df.columns, categorical_features=categorical_features)
            robustness['proba'].append(calibrators['uncal']['model'].predict_proba(X_test)[:,1])
            robustness['proba_va'].append(calibrators['va']['model'].predict_proba(X_test)[0][:,1])

            try:
                # print(f'{i}:',end='\t')
                # ce.set_random_state(i)
                tic = time.time()
                explanations = ce.explain_factual(X_test)
                ct = time.time()-tic
                rob_timer['ce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['ce'].append([f.feature_weights for f in explanations])

                # ce.set_random_state(i)
                tic = time.time()
                explanations = ce.explain_counterfactual(X_test)
                ct = time.time()-tic
                rob_timer['cce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['cce'].append([f.feature_weights for f in explanations])

                lime = LimeTabularExplainer(X_cal, feature_names=df.columns)
                model = calibrators['uncal']['model']
                tic = time.time()
                lime_exps_cl = []
                for instance in X_test:
                    exp = lime.explain_instance(instance, model.predict_proba)
                    lime_exps_cl.append(exp)
                ct = time.time()-tic
                rob_timer['lime'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                tmps = []
                for j in range(len(lime_exps_cl)):
                    tmp = np.zeros(no_of_features)
                    for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                        tmp[f[0]] = f[1]
                    tmps.append(tmp)
                robustness['lime'].append(tmps)

                model = calibrators['va']['model']
                tic = time.time()
                lime_exps_cl = []
                for instance in X_test:
                    exp = lime.explain_instance(instance, lambda x: model.predict_proba(x)[0])
                    lime_exps_cl.append(exp)
                ct = time.time()-tic
                rob_timer['lime_va'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                tmps = []
                for j in range(len(lime_exps_cl)):
                    tmp = np.zeros(no_of_features)
                    for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                        tmp[f[0]] = f[1]
                    tmps.append(tmp)
                robustness['lime_va'].append(tmps)

                shap = Explainer(lambda x: calibrators['uncal']['model'].predict_proba(x)[:,1], X_cal, \
                    feature_names=df.columns)
                # shap.random_state = i
                tic = time.time()
                explanations = shap(X_test)
                ct = time.time()-tic
                rob_timer['shap'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['shap'].append(explanations.values) # pylint: disable=no-member

                shap_va = Explainer(lambda x: calibrators['va']['model'].predict_proba(x)[0][:,1], X_cal, \
                    feature_names=df.columns)
                # shap.random_state = i
                tic = time.time()
                explanations = shap_va(X_test)
                ct = time.time()-tic
                rob_timer['shap_va'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['shap_va'].append(explanations.values) # pylint: disable=no-member


                i += 1
            except Exception as e: # pylint: disable=broad-exception-caught
                warnings.warn(f'Error: {e}')
            print('_',end='')
        print('')
        results[dataSet][alg]['robustness'] = robustness
        results[dataSet][alg]['rob_timer'] = rob_timer

    toc_data = time.time()
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )
    with open('evaluation/results_sota.pkl', 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()
debug_print(str(toc_data-tic_data),is_debug )
