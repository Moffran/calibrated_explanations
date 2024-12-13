# pylint: disable=all
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.utils import helper
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# MAIN PROGRAM
features_to_plot = 5

for dataSet in ["balance", "wave", "vowel", "cars", "steel", "heat", "cool", "user", "whole", "yeast" ]:#"iris", "tae", "image", "wineR", "glass", "vehicle", "wineW", "wine", "cmc", \
    
        # 

    fileName="data/Multiclass/multi/"+dataSet+".csv"
    df = pd.read_csv(fileName, sep=';')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    helper.make_directory(f"{dataSet}")

    if min(y) == 1:
        y = y - 1

    model = RandomForestClassifier()

    no_of_classes = len(np.unique(y))
    num_to_test = no_of_classes * 3
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]

    idx = list(range(no_of_instances))
    test_idx = [
        np.where(y == i)[0][: int(num_to_test / no_of_classes)]
        for i in range(no_of_classes)
    ]
    test_index = np.array(test_idx).flatten()

    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=42, stratify=y_train)

    print(dataSet, end=' - ', flush=True)

    model.fit(X_prop_train, y_prop_train)

    print('Model trained', end=' - ', flush=True)

    ce = CalibratedExplainer(model, X_cal, y_cal, \
                    feature_names=df.columns)
    factual_explanations = ce.explain_factual(X_test)
    for i in range(num_to_test):
        predicted = factual_explanations.get_explanation(i).prediction['classes']
        factual_explanations.plot(i, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_i{i}_c{y_test[i]}_p{predicted}.png")
        factual_explanations.plot(i, uncertainty=True, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_uncertainty_i{i}_c{y_test[i]}_p{predicted}.png")

    print('Factual Explanations done', end=' - ', flush=True)

    alternative_explanations = ce.explore_alternatives(X_test)
    for i in range(num_to_test):
        predicted = alternative_explanations.get_explanation(i).prediction['classes']
        alternative_explanations.plot(i, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_alternative_i{i}_c{y_test[i]}_p{predicted}.png")

    print('Alternative Explanations done', end=' - ', flush=True)

    cal_p = model.predict(X_cal)
    test_p = model.predict(X_test)

    ce = CalibratedExplainer(model, X_cal, y_cal, \
                    feature_names=df.columns, bins=cal_p)
    factual_explanations = ce.explain_factual(X_test, bins=test_p)
    for i in range(num_to_test):
        predicted = factual_explanations.get_explanation(i).prediction['classes']
        factual_explanations.plot(i, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_Mondrian_i{i}_c{y_test[i]}_p{predicted}.png")
        factual_explanations.plot(i, uncertainty=True, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_uncertainty_Mondrian_i{i}_c{y_test[i]}_p{predicted}.png")

    print('Mondrian Factual Explanations done', end=' - ', flush=True)

    alternative_explanations = ce.explore_alternatives(X_test, bins=test_p)
    for i in range(num_to_test):
        predicted = alternative_explanations.get_explanation(i).prediction['classes']
        alternative_explanations.plot(i, filter_top=features_to_plot, filename=f"{dataSet}/{dataSet}_alternative_Mondrian_i{i}_c{y_test[i]}_p{predicted}.png")

    print('Mondrian Alternative Explanations done')
