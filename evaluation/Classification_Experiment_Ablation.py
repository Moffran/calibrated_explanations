# pylint: disable=invalid-name, line-too-long
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
from calibrated_explanations import CalibratedExplainer

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------

test_size = .1 # number of test samples per dataset
is_debug = True
calibration_sizes = [0.1,0.2,0.4]
sample_percentiles = [[50],[33, 67],[25, 50, 75],[20, 40, 60, 80],[10, 20, 30, 40, 50, 60, 70, 80, 90]]


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
results['calibration_sizes'] = calibration_sizes
results['sample_percentiles'] = sample_percentiles
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
        
        trainCalX, testX, trainCalY, testY = train_test_split(X.values, y.values, test_size=test_size,random_state=42)
        trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=np.max(calibration_sizes),random_state=42)

        c2.fit(trainX,trainY)
        categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]

        
        ablation =  {'ce':{}, 'cce':{}, 'proba':{}, }
        abl_timer = {'ce':{}, 'cce':{}, }

        for cal_size in calibration_sizes:
            ablation['ce'][cal_size] = {}
            ablation['cce'][cal_size] = {}
            ablation['proba'][cal_size] = {}
            abl_timer['ce'][cal_size] = {}
            abl_timer['cce'][cal_size] = {}
            for sample_percentile in sample_percentiles:
                ablation['ce'][cal_size][str(sample_percentile)] = []
                ablation['cce'][cal_size][str(sample_percentile)] = []
                ablation['proba'][cal_size][str(sample_percentile)] = []
                abl_timer['ce'][cal_size][str(sample_percentile)] = []
                abl_timer['cce'][cal_size][str(sample_percentile)] = []

                cal_prop = int(np.max(calibration_sizes)/cal_size)
                calX_sample = calX[0::cal_prop,:]
                calY_sample = calY[0::cal_prop]
                ce = CalibratedExplainer(c2, calX_sample, calY_sample, \
                    feature_names=df.columns, categorical_features=categorical_features, sample_percentiles=sample_percentile)
                ablation['proba'][cal_size][str(sample_percentile)].append(c2.predict_proba(testX)[:,1])

                try:
                    # print(f'{i}:',end='\t')
                    tic = time.time()
                    factual_explanations = ce.explain_factual(testX)
                    ct = time.time()-tic
                    abl_timer['ce'][cal_size][str(sample_percentile)].append(ct)
                    # print(f'{ct:.1f}',end='\t')
                    ablation['ce'][cal_size][str(sample_percentile)].append([f.feature_weights for f in factual_explanations])

                    tic = time.time()
                    factual_explanation = ce.explain_counterfactual(testX)
                    ct = time.time()-tic
                    abl_timer['cce'][cal_size][str(sample_percentile)].append(ct)
                    # print(f'{ct:.1f}',end='\t')
                    ablation['cce'][cal_size][str(sample_percentile)].append([f.feature_weights for f in factual_explanations])

                except Exception as e: # pylint: disable=broad-exception-caught
                    warnings.warn(f'Error: {e}')
                # print('')

        results[dataSet][alg]['ablation'] = ablation
        results[dataSet][alg]['timer'] = abl_timer

    toc_data = time.time()
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )
    with open('evaluation/results_ablation.pkl', 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()
debug_print(str(toc_data-tic_data),is_debug )
