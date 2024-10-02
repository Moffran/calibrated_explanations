# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used in the paper introducing perturbed explanations to evaluate the stability and robustness of the explanations
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
from calibrated_explanations import CalibratedExplainer
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------

test_size = 1/4 # number of test samples per dataset
is_debug = True
# calibration_sizes = [0.1,0.2,0.4]
scale_factors = [1, 3, 5, 10]
severities = [0, 0.25, 0.5, 0.75, 1]
noise_type = ['uniform', 'gaussian']


descriptors = ['uncal','va',]#,'va'
Descriptors = {'uncal':'Uncal','va': 'VA'}
models = ['xGB','RF'] # ['xGB','RF','DT','SVM',] # 'NN',

# pylint: disable=line-too-long
datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1"}
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {}
results['severity'] = severities
results['noise_type'] = noise_type
results['scale_factor'] = scale_factors
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

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size,random_state=42)
        X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=1/3,random_state=42)

        c2.fit(X_prop_train,y_prop_train)
        categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]

        ablation =  {'ce':[], 'pce':{}, 'proba':[] }
        abl_timer = {'ce_init':[], 'ce_explain':[], 'pce_init':{}, 'pce_explain':{}, }

        tic = time.time()
        ce = CalibratedExplainer(c2, X_cal, y_cal, \
            feature_names=df.columns, categorical_features=categorical_features)
        ct = time.time()-tic
        abl_timer['ce_init'].append(ct/len(X_cal))

        tic = time.time()
        factual_explanations = ce.explain_factual(X_test)
        ct = time.time()-tic
        abl_timer['ce_explain'].append(ct/len(X_test))
        ablation['ce'].append([f.feature_weights for f in factual_explanations])
        ablation['proba'].append(c2.predict_proba(X_test)[:,1])

        for factor in scale_factors:
            ablation['pce'][factor] = {}
            abl_timer['pce_init'][factor] = {}
            abl_timer['pce_explain'][factor] = {}
            for severity in severities:
                ablation['pce'][factor][severity] = {}
                abl_timer['pce_init'][factor][severity] = {}
                abl_timer['pce_explain'][factor][severity] = {}
                for noise in noise_type:
                    ablation['pce'][factor][severity][noise] = []
                    abl_timer['pce_init'][factor][severity][noise] = []
                    abl_timer['pce_explain'][factor][severity][noise] = []

                    tic = time.time()
                    ce = CalibratedExplainer(c2, X_cal, y_cal, fast=True, \
                        feature_names=df.columns, categorical_features=categorical_features,
                        severity=severity, noise_type=noise, scale_factor=factor)
                    ct = time.time()-tic
                    abl_timer['pce_init'][factor][severity][noise].append(ct/len(X_cal))

                    # try:
                        # print(f'{i}:',end='\t')
                    tic = time.time()
                    fast_explanations = ce.explain_fast(X_test)
                    ct = time.time()-tic
                    abl_timer['pce_explain'][factor][severity][noise].append(ct/len(X_test))
                    # print(f'{ct:.1f}',end='\t')
                    ablation['pce'][factor][severity][noise].append([f.feature_weights for f in fast_explanations])


                # except Exception as e: # pylint: disable=broad-exception-caught
                #     warnings.warn(f'Error: {e}')
                # print('')

        results[dataSet][alg]['ablation'] = ablation
        results[dataSet][alg]['timer'] = abl_timer

    toc_data = time.time()
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )
    with open('evaluation/results_perturbed_ablation.pkl', 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()
debug_print(str(toc_data-tic_data),is_debug )
