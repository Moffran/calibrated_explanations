# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used in the introductory paper to evaluate the stability and robustness of the explanations
"""

import time
import warnings
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.utils.helper import transform_to_numeric
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# pylint: disable=missing-function-docstring, no-member
def wine():
    ds = fetch_openml(name="wine", version=7, as_frame=True, parser='pandas')

    X_ = ds.data.values.astype(float)
    y_ = (ds.target.values == 'True').astype(int)

    feature_names_ = ds.feature_names

    # Create a DataFrame from X
    df = pd.DataFrame(X_, columns=feature_names_)

    # Add the target column to the DataFrame
    target = 'target'
    df[target] = y_

    df, categorical_features_, categorical_labels_, target_labels_, _ = transform_to_numeric(df, target=target)
    X_ = df.drop(columns=[target]).values
    y_ = df[target].values
    return X_, y_, feature_names_, categorical_features_, categorical_labels_, target_labels_, 'wine', True, None

def iris():
    ds = fetch_openml(name="iris", version=1, as_frame=True, parser='auto')

    X_ = ds.data.values.astype(float)
    y_ = ds.target.values

    feature_names_ = ds.feature_names

    # Create a DataFrame from X
    df = pd.DataFrame(X_, columns=feature_names_)

    # Add the target column to the DataFrame
    target = 'target'
    df[target] = y_

    df, categorical_features_, categorical_labels_, target_labels_, _ = transform_to_numeric(df, target=target)
    X_ = df.drop(columns=[target]).values
    y_ = df[target].values
    return X_, y_, feature_names_, categorical_features_, categorical_labels_, target_labels_, 'iris', True, None

def housing():
    ds = fetch_openml(name="house_sales", version=3)

    X_ = ds.data.values.astype(float)
    y_ = ds.target.values/1000

    feature_names_ = ds.feature_names

    # Create a DataFrame from X
    df = pd.DataFrame(X_, columns=feature_names_)

    # Add the target column to the DataFrame
    target = 'target'
    df[target] = y_

    df, categorical_features_, categorical_labels_, target_labels_, _ = transform_to_numeric(df, target=target)
    X_ = df.drop(columns=[target]).values
    y_ = df[target].values
    return X_, y_, feature_names_, categorical_features_, categorical_labels_, target_labels_, 'housing', False, np.percentile(y, 50)

def glass():
    ds = 'glass'
    delimiter = ','

    fileName = f'data/Multiclass/{ds}.csv'
    df = pd.read_csv(fileName, delimiter=delimiter)
    target = 'Type'

    df = df.dropna()

    df, categorical_features_, categorical_labels_, target_labels_, _ = transform_to_numeric(df, target=target)
    X_ = df.drop(columns=[target]).values
    y_ = df[target].values
    feature_names_ = df.drop(columns=[target]).columns
    return X_, y_, feature_names_, categorical_features_, categorical_labels_, target_labels_, 'glass', True, None
# ------------------------------------------------------

test_size = 100 # number of test samples per dataset
is_debug = True
calibration_sizes = [500, 300, 100]

datasets = [wine, housing]#, glass, ]

tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {'calibration_sizes': calibration_sizes, 'test_size': test_size}
for loader in datasets:
    X, y, feature_names, categorical_features, categorical_labels, target_labels, dataset, classification, threshold = loader()
    tic_data = time.time()
    print(dataset)

    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]

    rfc = RandomForestClassifier()
    rfr = RandomForestRegressor()

    results[dataset] = {}
    model = WrapCalibratedExplainer(rfc if classification else rfr)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=max(calibration_sizes),random_state=42)

    model.fit(X_prop_train, y_prop_train)

    ablation =  {'total':{}, 'ensured':{}, 'counterfactual':{}, 'counterpotential':{}, 'semifactual':{}, 'semipotential':{}, 'superfactual':{}, 'superpotential':{}}
    abl_timer = {}

    for cal_size in calibration_sizes:
        for metric in ablation:
            ablation[metric][cal_size] = {'explore':[], 'conjugate':[]}
        abl_timer[cal_size] = {'explore':[], 'conjugate':[]}

        cal_prop = int(max(calibration_sizes)/cal_size)
        X_cal_sample = X_cal[0::cal_prop,:]
        y_cal_sample = y_cal[0::cal_prop]
        mode = 'classification' if classification else 'regression'
        model.calibrate(X_cal_sample, y_cal_sample, mode=mode,\
            feature_names=feature_names, categorical_features=categorical_features)

        tic = time.time()
        explanations = model.explore_alternatives(X_test, threshold=threshold)
        ct = time.time()-tic
        # print(f'{ct:.1f}',end='\t')
        for explanation in explanations:
            abl_timer[cal_size]['explore'].append(explanation.explain_time)
            cf = deepcopy(explanation).get_counter_explanations()
            cp = deepcopy(explanation).get_counter_explanations(include_potential=True)
            sef = deepcopy(explanation).get_semi_explanations()
            sep = deepcopy(explanation).get_semi_explanations(include_potential=True)
            suf = deepcopy(explanation).get_super_explanations()
            sup = deepcopy(explanation).get_super_explanations(include_potential=True)
            ens = deepcopy(explanation).get_ensured_explanations()
            ablation['total'][cal_size]['explore'].append(len(explanation))
            ablation['ensured'][cal_size]['explore'].append(len(ens))
            ablation['counterfactual'][cal_size]['explore'].append(len(cf))
            ablation['counterpotential'][cal_size]['explore'].append(len(cp)-len(cf))
            ablation['semifactual'][cal_size]['explore'].append(len(sef))
            ablation['semipotential'][cal_size]['explore'].append(len(sep)-len(sef))
            ablation['superfactual'][cal_size]['explore'].append(len(suf))
            ablation['superpotential'][cal_size]['explore'].append(len(sup)-len(suf))

        for explanation in explanations:
            tic = time.time()
            explanation.add_conjunctions()
            ct = time.time()-tic
            abl_timer[cal_size]['conjugate'].append(ct)
            cf = deepcopy(explanation).get_counter_explanations()
            cp = deepcopy(explanation).get_counter_explanations(include_potential=True)
            sef = deepcopy(explanation).get_semi_explanations()
            sep = deepcopy(explanation).get_semi_explanations(include_potential=True)
            suf = deepcopy(explanation).get_super_explanations()
            sup = deepcopy(explanation).get_super_explanations(include_potential=True)
            ens = deepcopy(explanation).get_ensured_explanations()
            ablation['total'][cal_size]['conjugate'].append(len(explanation))
            ablation['ensured'][cal_size]['conjugate'].append(len(ens))
            ablation['counterfactual'][cal_size]['conjugate'].append(len(cf))
            ablation['counterpotential'][cal_size]['conjugate'].append(len(cp)-len(cf))
            ablation['semifactual'][cal_size]['conjugate'].append(len(sef))
            ablation['semipotential'][cal_size]['conjugate'].append(len(sep)-len(sef))
            ablation['superfactual'][cal_size]['conjugate'].append(len(suf))
            ablation['superpotential'][cal_size]['conjugate'].append(len(sup)-len(suf))

        # except Exception as e: # pylint: disable=broad-exception-caught
        #     warnings.warn(f'Error: {e}')
        # print('')

    results[dataset]['ablation'] = ablation
    results[dataset]['timer'] = abl_timer

    toc_data = time.time()
    debug_print(dataset + ': ' +str(toc_data-tic_data),is_debug )
    with open('evaluation/secret/results_ensured_ablation.pkl', 'wb') as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()
debug_print(str(toc_data-tic_data),is_debug )
