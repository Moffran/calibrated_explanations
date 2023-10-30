"""

"""

# from experiment_utils import (ece, clip)

import copy
import time
import warnings
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.calibration import calibration_curve
# from lime.lime_tabular import LimeTabularExplainer 
# from shap import Explainer
# from sklearn.preprocessing import StandardScaler
# from sklearn.calibration import CalibratedClassifierCV
from venn_abers import VennAbers
from calibrated_explanations import CalibratedExplainer
# from lipschitz_metric import lipschitz_metric, get_perturbations, explainer_func_wrapper
# from tqdm import tqdm



def bin_total(y_true, y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1
    return np.bincount(binids, minlength=len(bins))

def ece(y_true, y_prob, fop, mpv, n_bins=10):
    bins = bin_total(y_true, y_prob, n_bins)
    bins = bins[bins != 0]
    w = bins / np.sum(bins)
    return np.sum(w * abs(fop - mpv))

def clip(val, min_=0, max_=1):
    if len(val) == 1:
        return min_ if val < min_ else max_ if val > max_ else val
    return [min_ if v < min_ else max_ if v > max_ else v for v in val]

# -------------------------------------------------------

def debug_print(message, debug=True):
    if debug:
        print(message)
        
# ------------------------------------------------------

test_size = 20 # Antal foldar ändrade från 10
number_of_bins = 10
eval_matrix = []
is_debug = True
num_rep = 30

descriptors = ['uncal','va',]#,'va'
Descriptors = {'uncal':'Uncal','va': 'VA'} 
models = ['xGB','RF'] # ['xGB','RF','DT','SVM',] # 'NN',
explainers = ['ce','lime','shap'] #  ---------------------------->tagit bort  , sist

datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1"} # -------------------> tagit bort  , efter sista termen
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
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
    Xn, y = df.drop('Y',axis=1), df['Y'] # Dela upp datamängden i inputattribut och targetattribut

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15) # Changed from min_leaf=4
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True) # Skala även input
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
        trainCalX, testX, trainCalY, testY = train_test_split(X.values, y.values, test_size=test_size,random_state=42)
        trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
        
        c2.fit(trainX,trainY)

        calibrators['uncal']['model'] = c2
        if descriptors.__contains__('va'):                        
            calibrators['va']['model'] = VennAbers()
            calibrators['va']['model'].fit(c2.predict_proba(calX), calY)
        calibrators['data'] = {'trainX':trainX,'trainY':trainY,'calX':calX,'calY':calY,'testX':testX,'testY':testY,}

        # debug_print(dataSet + ' ' + alg , is_debug)
        np.random.seed(1337)
        categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
        
        ce = CalibratedExplainer(c2, calX, calY, \
            feature_names=df.columns, categorical_features=categorical_features)

        stability =  {'ce':[], 'cce':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]} 
        stab_timer = {'ce':[], 'cce':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        robustness = {'ce':[], 'cce':[], 'proba':[]}#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]} 
        rob_timer =  {'ce':[], 'cce':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        i = 0
        while i < num_rep:
            try:
                # print(f'{i}:',end='\t')
                
                ce.set_random_state(i)
                tic = time.time()
                factual_explanations = ce.explain_factual(testX)
                ct = time.time()-tic
                stab_timer['ce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                stability['ce'].append([f.feature_weights for f in factual_explanations])
                
                ce.set_random_state(i)
                tic = time.time()
                factual_explanation = ce.explain_counterfactual(testX)
                ct = time.time()-tic
                stab_timer['cce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                stability['cce'].append([f.feature_weights for f in factual_explanations])
                i += 1
            except Exception as e:
                warnings.warn(f'Error: {e}')
            # print('')
            
        
        results[dataSet][alg]['stability'] = stability
        results[dataSet][alg]['stab_timer'] = stab_timer
        
        i = 0
        while i < num_rep:
            np.random.seed(i)
            if alg == 'xGB':
                c2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='logloss')
            else:
                c2 = RandomForestClassifier(n_estimators=100)
            trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
            
            c2.fit(trainX,trainY)
            ce = CalibratedExplainer(c2, calX, calY, \
                feature_names=df.columns, categorical_features=categorical_features)
            robustness['proba'].append(c2.predict_proba(testX)[:,1])

            try:
                # print(f'{i}:',end='\t')
                
                ce.set_random_state(i)
                tic = time.time()
                factual_explanations = ce.explain_factual(testX)
                ct = time.time()-tic
                rob_timer['ce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['ce'].append([f.feature_weights for f in factual_explanations])
                
                ce.set_random_state(i)
                tic = time.time()
                factual_explanation = ce.explain_counterfactual(testX)
                ct = time.time()-tic
                rob_timer['cce'].append(ct)
                # print(f'{ct:.1f}',end='\t')
                robustness['cce'].append([f.feature_weights for f in factual_explanations])
                i += 1
            except Exception as e:
                warnings.warn(f'Error: {e}')
            # print('')

        results[dataSet][alg]['robustness'] = robustness
        results[dataSet][alg]['rob_timer'] = rob_timer
    

    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  
    pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 