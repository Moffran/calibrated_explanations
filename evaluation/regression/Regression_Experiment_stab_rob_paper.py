# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used in the introductory paper to evaluate the stability and robustness of the explanations
"""
import time
import warnings
import pickle
import numpy as np
import pandas as pd
#
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from crepes.extras import DifficultyEstimator

from calibrated_explanations import CalibratedExplainer

# Ignore all warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------

test_size = 10 # number of test samples per dataset
is_debug = True
num_rep = 100

descriptors = ['uncal','va',]#,'va'
Descriptors = {'uncal':'Uncal','va': 'VA'}
models = ['RF'] # ['xGB','RF','DT','SVM',] # 'NN',

# pylint: disable=line-too-long
datasets = {1:"housing"}
klara = [1]
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {}
results['num_rep'] = num_rep
results['test_size'] = test_size
try:
    dataSet = datasets[1]

    tic_data = time.time()
    delimiter = ';'
    categorical_labels = {8: {0: 'INLAND', 1: 'NEAR BAY', 2: '<1H OCEAN', 3: 'NEAR OCEAN', 4: 'ISLAND'}}

    fileName = 'data/reg/' + dataSet + '.csv'
    df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)
    target = 'median_house_value'
    df.dropna(inplace=True)
    Xn, y = df.drop(target,axis=1), df[target]
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_normalized = np.squeeze(scaler.fit_transform(y.values.reshape(-1, 1)))

    feature_names = df.drop(target,axis=1).columns

    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    r1 = RandomForestRegressor(n_estimators=100, oob_score=True)
    r2 = RandomForestRegressor(n_estimators=100, oob_score=True)

    model_dict = {'RF':(r1,r2,"RF",Xn)}#,'NN': (a1,a2,"NN",Xn)
    model_struct = [model_dict[model] for model in models]
    results[dataSet] = {}
    for c1, model, alg, X in model_struct:
        tic_algorithm = time.time()
        debug_print(dataSet+' '+alg)
        results[dataSet][alg] = {}

        X_trainCal, X_test, y_trainCal, y_test = train_test_split(X.values, y_normalized, test_size=test_size,random_state=42)
        X_train, X_cal, y_train, y_cal = train_test_split(X_trainCal, y_trainCal, test_size=500,random_state=42)
        X_train = X_train[:1000]
        y_train = y_train[:1000]

        model.fit(X_train,y_train)

        p_test = model.predict(X_test)
        p_cal = model.predict(X_cal)
        r_test = y_test - p_test
        r_cal = y_cal - p_cal

        # de_dist = DifficultyEstimator().fit(X=X_train, scaler=True)
        # de_std  = DifficultyEstimator().fit(X=X_train, y=y_train, scaler=True)
        # de_abs  = DifficultyEstimator().fit(X=X_train, residuals=y_train - model.oob_prediction_, scaler=True)
        de_var  = DifficultyEstimator().fit(X=X_train, learner=model, scaler=True)

        np.random.seed(1337)
        # categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]

        stability =  {'ce':[], 'cce':[], 'ce_var':[], 'cce_var':[], 'pce':[], 'pcce':[], 'pce_var':[], 'pcce_var':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        stab_timer = {'ce':[], 'cce':[], 'ce_var':[], 'cce_var':[], 'pce':[], 'pcce':[], 'pce_var':[], 'pcce_var':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        robustness = {'ce':[], 'cce':[], 'ce_var':[], 'cce_var':[], 'pce':[], 'pcce':[], 'pce_var':[], 'pcce_var':[], 'predict':[]}#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        rob_timer =  {'ce':[], 'cce':[], 'ce_var':[], 'cce_var':[], 'pce':[], 'pcce':[], 'pce_var':[], 'pcce_var':[], }#'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
        i = 0
        while i < num_rep:
            print(f'{i+1}:',end='\n', flush=True)
            ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression', random_state=i)
            # try:
            tic = time.time()
            factual_explanations = ce.explain_factual(X_test)
            ct = time.time()-tic
            stab_timer['ce'].append(ct)
            print(f' f{ct:.1f}',end=' ', flush=True)
            stability['ce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanation = ce.explain_counterfactual(X_test)
            ct = time.time()-tic
            stab_timer['cce'].append(ct)
            print(f' c{ct:.1f}',end=' ', flush=True)
            stability['cce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanations = ce.explain_factual(X_test, threshold=0.5)
            ct = time.time()-tic
            stab_timer['pce'].append(ct)
            print(f' pf{ct:.1f}',end=' ', flush=True)
            stability['pce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanation = ce.explain_counterfactual(X_test, threshold=0.5)
            ct = time.time()-tic
            stab_timer['pcce'].append(ct)
            print(f' pc{ct:.1f}',end='\n', flush=True)
            stability['pcce'].append([f.feature_weights for f in factual_explanations])

            # print(f'no normalization:{}',end=' ')
            for norm in ['_var']:
                # if norm == '_dist':
                #     de = de_dist
                # elif norm == '_std':
                #     de = de_std
                # elif norm == '_abs':
                #     de = de_abs
                # elif norm == '_var':
                de = de_var
                ce.set_difficulty_estimator(de)

                tic = time.time()
                ce.explain_factual(X_test, threshold=0.5)
                ct = time.time()-tic
                stab_timer['ce'+norm].append(ct)
                print(f'{norm[1]}f{ct:.1f}',end=' ', flush=True)
                stability['ce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_counterfactual(X_test, threshold=0.5)
                ct = time.time()-tic
                stab_timer['cce'+norm].append(ct)
                print(f'{norm[1]}c{ct:.1f}',end=' ', flush=True)
                stability['cce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_factual(X_test)
                ct = time.time()-tic
                stab_timer['pce'+norm].append(ct)
                print(f'{norm[1]}pf{ct:.1f}',end=' ', flush=True)
                stability['pce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_counterfactual(X_test)
                ct = time.time()-tic
                stab_timer['pcce'+norm].append(ct)
                print(f'{norm[1]}pc{ct:.1f}',end='\n', flush=True)
                stability['pcce'+norm].append([f.feature_weights for f in factual_explanations])
            # print(f'',end='\n', flush=True)
            i += 1
            # except Exception as e: # pylint: disable=broad-exception-caught
            #     warnings.warn(f'Error: {e}')
            # print('')

        results[dataSet][alg]['stability'] = stability
        results[dataSet][alg]['stab_timer'] = stab_timer
        with open('evaluation/regression/results_stab_paper.pkl', 'wb') as f:
            pickle.dump(results, f)

        i = 0
        while i < num_rep:
            print(f'{i+1}:',end='\n', flush=True)
            np.random.seed = i
            model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=i)
            X_train, X_cal, y_train, y_cal = train_test_split(X_trainCal, y_trainCal, test_size=500,random_state=i)
            X_train = X_train[:1000]
            y_train = y_train[:1000]

            model.fit(X_train, y_train)
            # de_dist = DifficultyEstimator().fit(X=X_train, scaler=True)
            # de_std  = DifficultyEstimator().fit(X=X_train, y=y_train, scaler=True)
            # de_abs  = DifficultyEstimator().fit(X=X_train, residuals=y_train - model.oob_prediction_, scaler=True)
            de_var  = DifficultyEstimator().fit(X=X_train, learner=model, scaler=True)

            ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression',random_state=i)
            robustness['predict'].append(model.predict(X_test))

            # try:
            tic = time.time()
            factual_explanations = ce.explain_factual(X_test)
            ct = time.time()-tic
            rob_timer['ce'].append(ct)
            print(f' f{ct:.1f}',end=' ', flush=True)
            robustness['ce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanation = ce.explain_counterfactual(X_test)
            ct = time.time()-tic
            rob_timer['cce'].append(ct)
            print(f' c{ct:.1f}',end=' ', flush=True)
            robustness['cce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanations = ce.explain_factual(X_test, threshold=0.5)
            ct = time.time()-tic
            rob_timer['pce'].append(ct)
            print(f' pf{ct:.1f}',end=' ', flush=True)
            robustness['pce'].append([f.feature_weights for f in factual_explanations])

            tic = time.time()
            factual_explanation = ce.explain_counterfactual(X_test, threshold=0.5)
            ct = time.time()-tic
            rob_timer['pcce'].append(ct)
            print(f' pc{ct:.1f}',end='\n', flush=True)
            robustness['pcce'].append([f.feature_weights for f in factual_explanations])

            for norm in ['_var']:
                # if norm == '_dist':
                #     de = de_dist
                # elif norm == '_std':
                #     de = de_std
                # elif norm == '_abs':
                #     de = de_abs
                # elif norm == '_var':
                de = de_var
                ce.set_difficulty_estimator(de)

                tic = time.time()
                ce.explain_factual(X_test, threshold=0.5)
                ct = time.time()-tic
                rob_timer['ce'+norm].append(ct)
                print(f'{norm[1]}f{ct:.1f}',end=' ', flush=True)
                robustness['ce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_counterfactual(X_test, threshold=0.5)
                ct = time.time()-tic
                rob_timer['cce'+norm].append(ct)
                print(f'{norm[1]}c{ct:.1f}',end=' ', flush=True)
                robustness['cce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_factual(X_test)
                ct = time.time()-tic
                rob_timer['pce'+norm].append(ct)
                print(f'{norm[1]}pf{ct:.1f}',end=' ', flush=True)
                robustness['pce'+norm].append([f.feature_weights for f in factual_explanations])

                tic = time.time()
                ce.explain_counterfactual(X_test)
                ct = time.time()-tic
                rob_timer['pcce'+norm].append(ct)
                print(f'{norm[1]}pc{ct:.1f}',end='\n', flush=True)
                robustness['pcce'+norm].append([f.feature_weights for f in factual_explanations])
            i += 1
            # except Exception as e: # pylint: disable=broad-exception-caught
            #     warnings.warn(f'Error: {e}')
            # print('')

        results[dataSet][alg]['robustness'] = robustness
        results[dataSet][alg]['rob_timer'] = rob_timer

    toc_data = time.time()
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )
    with open('evaluation/regression/results_rob_paper.pkl', 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))
    toc_all = time.time()
    debug_print(str(toc_data-tic_data),is_debug )
except Exception as e:
    print(f'Error: {e}')
