# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used in the introductory paper to evaluate the stability and robustness of the explanations
"""
import time
import warnings
import pickle
import os
from click import progressbar
import numpy as np
import pandas as pd
import shap
#
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from crepes import ConformalPredictiveSystem
from crepes.extras import DifficultyEstimator

from lime.lime_tabular import LimeTabularExplainer
import sys

# Add the project root to the Python path
sys.path.append(r'C:\Users\loftuw\Documents\Github\calibrated_explanations')
sys.path.append(r'C:\Users\loftuw\Documents\Github\calibrated_explanations\src')


from calibrated_explanations import CalibratedExplainer

# Ignore all warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# -------------------------------------------------------

def explain_shap(shap_explainer, data):
    shap_values = shap_explainer(data)
    return [{'predict':sv} for sv in shap_values.values]

#pylint: disable=redefined-outer-name
def explain_lime(lime_explainer, predictor, data):
    feature_weights = []
    for x in data:
        exp = lime_explainer.explain_instance(x, predictor, num_features=data.shape[1])
        features = [exp.local_exp[1][f][0] for f in range(len(exp.local_exp[1]))]
        weights = np.zeros(len(features))
        for i, f in enumerate(features):
            weights[f] = exp.local_exp[1][i][1]
        feature_weights.append({'predict':weights})
    return feature_weights

# ------------------------------------------------------

test_size = 100 # number of test samples per dataset
cal_size = 200 # number of calibration samples per dataset
is_debug = True
num_rep = 5
normalizations = ['']# ['', '_dist', '_std', '_abs', '_var']#['', '_var']#
resultfile = 'evaluation/secret/results_regression_test.pkl'

descriptors = ['uncal','va',]#,'va'
Descriptors = {'uncal':'Uncal','va': 'VA'}
models = ['RF'] # ['xGB','RF','DT','SVM',] # 'NN',
setups = ['lime','shap','ce','pce','fce','pfce']



path = 'data/reg'
if isinstance(path,str) and os.path.isdir(path):
    filenames = [os.path.join(path,file) for file in os.listdir(os.path.join(os.getcwd(),path)) if file.endswith(".txt")]
    datasets = [file.rsplit( ".", 1 )[ 0 ] for file in os.listdir(os.path.join(os.getcwd(),path)) if file.endswith(".txt")]

# pylint: disable=line-too-long
# datasets = {1:"housing"}
klara = [1]
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {}
results['num_rep'] = num_rep
results['test_size'] = test_size

np.random.seed(1337)
for dataSet, filename in zip(datasets, filenames):

    tic_data = time.time()
    delimiter = ';'
    categorical_labels = {8: {0: 'INLAND', 1: 'NEAR BAY', 2: '<1H OCEAN', 3: 'NEAR OCEAN', 4: 'ISLAND'}}

    df = pd.read_csv(filename, usecols = lambda x: False if (x == "ID" or x.startswith("IGNORE")) else True)
    target = 'REGRESSION'
    df.dropna(inplace=True)
    Xn, y = df.drop(target,axis=1), df[target]
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_normalized = np.squeeze(scaler.fit_transform(y.values.reshape(-1, 1)))

    feature_names = df.drop(target,axis=1).columns

    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]
    if no_of_instances < 800: # skip datasets with less than 800 instances, as 100 + 200 + 500 = 800
        continue

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
        X_train, X_cal, y_train, y_cal = train_test_split(X_trainCal, y_trainCal, test_size=cal_size,random_state=42)
        # X_train = X_train[:1000]
        # y_train = y_train[:1000]

        model.fit(X_train,y_train)

        p_test = model.predict(X_test)
        p_cal = model.predict(X_cal)
        r_test = y_test - p_test
        r_cal = y_cal - p_cal

        de_none = None
        de_dist = DifficultyEstimator().fit(X=X_train[:500], scaler=True)
        de_std  = DifficultyEstimator().fit(X=X_train[:500], y=y_train[:500], scaler=True)
        de_abs  = DifficultyEstimator().fit(X=X_train[:500], residuals=y_train[:500] - model.oob_prediction_[:500], scaler=True)
        de_var  = DifficultyEstimator().fit(X=X_train[:500], learner=model, scaler=True)

        s_cal_dist = de_dist.apply(X_cal)
        s_cal_std = de_std.apply(X_cal)
        s_cal_abs = de_abs.apply(X_cal)
        s_cal_var = de_var.apply(X_cal)

        cps_none = ConformalPredictiveSystem().fit(residuals=r_cal)
        cps_dist = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_dist)
        cps_std = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_std)
        cps_abs = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_abs)
        cps_var = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_var)

        # pylint: disable=unnecessary-lambda-assignment, cell-var-from-loop
        predictor_none = lambda x: np.mean(cps_none.predict(model.predict(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
        predictor_dist = lambda x: np.mean(cps_dist.predict(model.predict(x), sigmas=de_dist.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
        predictor_std = lambda x: np.mean(cps_std.predict(model.predict(x), sigmas=de_std.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
        predictor_abs = lambda x: np.mean(cps_abs.predict(model.predict(x), sigmas=de_abs.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
        predictor_var = lambda x: np.mean(cps_var.predict(model.predict(x), sigmas=de_var.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)

        # categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]

        stability =  {'lime_base':[], 'shap_base':[]}
        stab_timer = {'lime_base':[], 'shap_base':[]}
        robustness = {'lime_base':[], 'shap_base':[],'predict':[]}
        rob_timer =  {'lime_base':[], 'shap_base':[]}
        for setup in setups:
            for norm in normalizations:
                stability[setup+norm] = []
                stab_timer[setup+norm] = []
                robustness[setup+norm] = []
                rob_timer[setup+norm] = []
        i = 0
        while i < num_rep:
            print(f'Stability {i+1} - {len(X_test)}:',end='\n', flush=True)
            ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression', random_state=i)
            fce = CalibratedExplainer(model, X_cal, y_cal, mode='regression', random_state=i, fast=True)
            # print(f'no normalization:{}',end=' ')
            se = shap.Explainer(lambda x: model.predict(x), X_cal, seed=i) # pylint: disable=unnecessary-lambda
            explain_shap(se, X_test[:1]) # initialization call, to avoid overhead in first call
            le = LimeTabularExplainer(X_cal, mode='regression', random_state=i)

            tic = time.time()
            explanations = explain_shap(se, X_test)
            ct = time.time()-tic
            stab_timer['shap_base'].append(ct/len(X_test))
            print(f'bs{ct/len(X_test):.3f}',end=' ', flush=True)
            stability['shap_base'].append(explanations)

            tic = time.time()
            explanations = explain_lime(le, lambda x: model.predict(x), X_test) # pylint: disable=unnecessary-lambda
            ct = time.time()-tic
            stab_timer['lime_base'].append(ct/len(X_test))
            print(f'bl{ct/len(X_test):.3f}',end='\n', flush=True)
            stability['lime_base'].append(explanations)

            for norm in normalizations:
                if norm == '_dist':
                    ce.set_difficulty_estimator(de_dist)
                    predictor = predictor_dist
                    letter = 'd'
                elif norm == '_std':
                    ce.set_difficulty_estimator(de_std)
                    predictor = predictor_std
                    letter = 's'
                elif norm == '_abs':
                    ce.set_difficulty_estimator(de_abs)
                    predictor = predictor_abs
                    letter = 'a'
                elif norm == '_var':
                    ce.set_difficulty_estimator(de_var)
                    predictor = predictor_var
                    letter = 'v'
                else:
                    letter = ' '
                    predictor = predictor_none
                se = shap.Explainer(predictor, X_cal, seed=i)
                explain_shap(se, X_test[:1]) # initialization call, to avoid overhead in first call

                tic = time.time()
                explanations = explain_shap(se, X_test)
                ct = time.time()-tic
                stab_timer['shap'+norm].append(ct/len(X_test))
                print(f'{letter}s{ct/len(X_test):.3f}',end=' ', flush=True)
                stability['shap'+norm].append(explanations)

                tic = time.time()
                explanations = explain_lime(le, predictor, X_test) # pylint: disable=unnecessary-lambda
                ct = time.time()-tic
                stab_timer['lime'+norm].append(ct/len(X_test))
                print(f'{letter}l{ct/len(X_test):.3f}',end=' ', flush=True)
                stability['lime'+norm].append(explanations)

                tic = time.time()
                explanations = ce.explain_factual(X_test)
                ct = time.time()-tic
                stab_timer['ce'+norm].append(ct/len(X_test))
                print(f'{letter}ce{ct/len(X_test):.3f}',end=' ', flush=True)
                stability['ce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = ce.explain_factual(X_test, threshold=0.5)
                ct = time.time()-tic
                stab_timer['pce'+norm].append(ct/len(X_test))
                print(f'{letter}pce{ct/len(X_test):.3f}',end=' ', flush=True)
                stability['pce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = fce.explain_fast(X_test)
                ct = time.time()-tic
                stab_timer['fce'+norm].append(ct/len(X_test))
                print(f'{letter}fce{ct/len(X_test):.4f}',end=' ', flush=True)
                stability['fce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = fce.explain_fast(X_test, threshold=0.5)
                ct = time.time()-tic
                stab_timer['pfce'+norm].append(ct/len(X_test))
                print(f'{letter}pfce{ct/len(X_test):.4f}',end='\n', flush=True)
                stability['pfce'+norm].append([f.feature_weights for f in explanations])
            # print(f'',end='\n', flush=True)
            i += 1
            # except Exception as e: # pylint: disable=broad-exception-caught
            #     warnings.warn(f'Error: {e}')
            # print('')

        results[dataSet][alg]['stability'] = stability
        results[dataSet][alg]['stab_timer'] = stab_timer
        with open(resultfile, 'wb') as f:
            pickle.dump(results, f)

        i = 0
        while i < num_rep:
            print(f'Robustness {i+1} - {len(X_test)}:',end='\n', flush=True)
            np.random.seed = i
            model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=i)
            X_train, X_cal, y_train, y_cal = train_test_split(X_trainCal, y_trainCal, test_size=cal_size,random_state=i)
            # X_train = X_train[:1000]
            # y_train = y_train[:1000]

            model.fit(X_train, y_train)
            de_dist = DifficultyEstimator().fit(X=X_train[:500], scaler=True)
            de_std  = DifficultyEstimator().fit(X=X_train[:500], y=y_train[:500], scaler=True)
            de_abs  = DifficultyEstimator().fit(X=X_train[:500], residuals=y_train[:500] - model.oob_prediction_[:500], scaler=True)
            de_var  = DifficultyEstimator().fit(X=X_train[:500], learner=model, scaler=True)

            s_cal_dist = de_dist.apply(X_cal)
            s_cal_std = de_std.apply(X_cal)
            s_cal_abs = de_abs.apply(X_cal)
            s_cal_var = de_var.apply(X_cal)

            cps_none = ConformalPredictiveSystem().fit(residuals=r_cal)
            cps_dist = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_dist)
            cps_std = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_std)
            cps_abs = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_abs)
            cps_var = ConformalPredictiveSystem().fit(residuals=r_cal, sigmas=s_cal_var)

            # pylint: disable=unnecessary-lambda-assignment, cell-var-from-loop
            predictor_none = lambda x: np.mean(cps_none.predict(model.predict(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
            predictor_dist = lambda x: np.mean(cps_dist.predict(model.predict(x), sigmas=de_dist.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
            predictor_std = lambda x: np.mean(cps_std.predict(model.predict(x), sigmas=de_std.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
            predictor_abs = lambda x: np.mean(cps_abs.predict(model.predict(x), sigmas=de_abs.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)
            predictor_var = lambda x: np.mean(cps_var.predict(model.predict(x), sigmas=de_var.apply(x), lower_percentiles=[50], higher_percentiles=[50]), axis=1)

            ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression',random_state=i)
            fce = CalibratedExplainer(model, X_cal, y_cal, mode='regression',random_state=i, fast=True)
            robustness['predict'].append(model.predict(X_test))

            se = shap.Explainer(lambda x: model.predict(x), X_cal, seed=i) # pylint: disable=unnecessary-lambda
            explain_shap(se, X_test[:1]) # initialization call, to avoid overhead in first call
            le = LimeTabularExplainer(X_cal, mode='regression', random_state=i)

            tic = time.time()
            explanations = explain_shap(se, X_test)
            ct = time.time()-tic
            rob_timer['shap_base'].append(ct/len(X_test))
            print(f'bs{ct/len(X_test):.3f}',end=' ', flush=True)
            robustness['shap_base'].append(explanations)

            tic = time.time()
            explanations = explain_lime(le, lambda x: model.predict(x), X_test) # pylint: disable=unnecessary-lambda
            ct = time.time()-tic
            rob_timer['lime_base'].append(ct/len(X_test))
            print(f'bl{ct/len(X_test):.3f}',end='\n', flush=True)
            robustness['lime_base'].append(explanations)

            # try:

            for norm in normalizations:
                if norm == '_dist':
                    ce.set_difficulty_estimator(de_dist)
                    predictor = predictor_dist
                    letter = 'd'
                elif norm == '_std':
                    ce.set_difficulty_estimator(de_std)
                    predictor = predictor_std
                    letter = 's'
                elif norm == '_abs':
                    ce.set_difficulty_estimator(de_abs)
                    predictor = predictor_abs
                    letter = 'a'
                elif norm == '_var':
                    ce.set_difficulty_estimator(de_var)
                    predictor = predictor_var
                    letter = 'v'
                else:
                    letter = ' '
                    predictor = predictor_none
                se = shap.Explainer(predictor, X_cal, seed=i)
                explain_shap(se, X_test[:1]) # initialization call, to avoid overhead in first call

                tic = time.time()
                explanations = explain_shap(se, X_test)
                ct = time.time()-tic
                rob_timer['shap'+norm].append(ct/len(X_test))
                print(f'{letter}s{ct/len(X_test):.3f}',end=' ', flush=True)
                robustness['shap'+norm].append(explanations)

                tic = time.time()
                explanations = explain_lime(le, predictor, X_test) # pylint: disable=unnecessary-lambda
                ct = time.time()-tic
                rob_timer['lime'+norm].append(ct/len(X_test))
                print(f'{letter}l{ct/len(X_test):.3f}',end=' ', flush=True)
                robustness['lime'+norm].append(explanations)

                tic = time.time()
                explanations = ce.explain_factual(X_test)
                ct = time.time()-tic
                rob_timer['ce'+norm].append(ct/len(X_test))
                print(f'{letter}ce{ct/len(X_test):.3f}',end=' ', flush=True)
                robustness['ce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = ce.explain_factual(X_test, threshold=0.5)
                ct = time.time()-tic
                rob_timer['pce'+norm].append(ct/len(X_test))
                print(f'{letter}pce{ct/len(X_test):.3f}',end=' ', flush=True)
                robustness['pce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = fce.explain_fast(X_test)
                ct = time.time()-tic
                rob_timer['fce'+norm].append(ct/len(X_test))
                print(f'{letter}fce{ct/len(X_test):.4f}',end=' ', flush=True)
                robustness['fce'+norm].append([f.feature_weights for f in explanations])

                tic = time.time()
                explanations = fce.explain_fast(X_test, threshold=0.5)
                ct = time.time()-tic
                rob_timer['pfce'+norm].append(ct/len(X_test))
                print(f'{letter}pfce{ct/len(X_test):.4f}',end='\n', flush=True)
                robustness['pfce'+norm].append([f.feature_weights for f in explanations])
            i += 1
            # except Exception as e: # pylint: disable=broad-exception-caught
            #     warnings.warn(f'Error: {e}')
            # print('')

        results[dataSet][alg]['robustness'] = robustness
        results[dataSet][alg]['rob_timer'] = rob_timer

    toc_data = time.time()
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )
    with open(resultfile, 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))
    toc_all = time.time()
    debug_print(str(toc_data-tic_data),is_debug )
# except Exception as e: # pylint: disable=broad-exception-caught
#     print(f'Error: {e}')
