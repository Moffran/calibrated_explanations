# pylint: disable=all
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import CalibratedExplainer, utils
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# MAIN PROGRAM
features_to_plot = 5

for dataSet in ["balance", "wave", "vowel", "cars", "steel", "heat", "cool", "user", "whole", "yeast" ]:#"iris", "tae", "image", "wineR", "glass", "vehicle", "wineW", "wine", "cmc", \
    
        # 

    fileName="data/Multiclass/multi/"+dataSet+".csv"
    df = pd.read_csv(fileName, sep=';')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    utils.make_directory(f"{dataSet}")

    if min(y) == 1:
        y = y - 1

    model = RandomForestClassifier()

    no_of_classes = len(np.unique(y))
    num_to_test = no_of_classes * 3
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]

    test_idx = []
    idx = list(range(no_of_instances))
    for i in range(no_of_classes):
        test_idx.append(np.where(y == i)[0][0:int(num_to_test/no_of_classes)])
    test_index = np.array(test_idx).flatten()

    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)
    trainCalX, testX = X[train_index,:], X[test_index,:]
    trainCalY, testY = y[train_index], y[test_index]
    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42, stratify=trainCalY)

    print(dataSet, end=' - ', flush=True)

    model.fit(trainX, trainY)

    print('Model trained', end=' - ', flush=True)

    ce = CalibratedExplainer(model, calX, calY, \
                    feature_names=df.columns)
    factual_explanations = ce.explain_factual(testX)
    for i in range(num_to_test):
        predicted = factual_explanations.get_explanation(i).prediction['classes']
        factual_explanations.plot_explanation(i, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_i{i}_c{testY[i]}_p{predicted}.png")
        factual_explanations.plot_explanation(i, uncertainty=True, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_uncertainty_i{i}_c{testY[i]}_p{predicted}.png")

    print('Factual Explanations done', end=' - ', flush=True)

    counterfactual_explanations = ce.explain_counterfactual(testX)
    for i in range(num_to_test):
        predicted = counterfactual_explanations.get_explanation(i).prediction['classes']
        counterfactual_explanations.plot_explanation(i, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_counterfactual_i{i}_c{testY[i]}_p{predicted}.png")

    print('Counterfactual Explanations done', end=' - ', flush=True)
    
    cal_p = model.predict(calX)
    test_p = model.predict(testX)

    ce = CalibratedExplainer(model, calX, calY, \
                    feature_names=df.columns, bins=cal_p)
    factual_explanations = ce.explain_factual(testX, bins=test_p)
    for i in range(num_to_test):
        predicted = factual_explanations.get_explanation(i).prediction['classes']
        factual_explanations.plot_explanation(i, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_Mondrian_i{i}_c{testY[i]}_p{predicted}.png")
        factual_explanations.plot_explanation(i, uncertainty=True, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_factual_uncertainty_Mondrian_i{i}_c{testY[i]}_p{predicted}.png")

    print('Mondrian Factual Explanations done', end=' - ', flush=True)

    counterfactual_explanations = ce.explain_counterfactual(testX, bins=test_p)
    for i in range(num_to_test):
        predicted = counterfactual_explanations.get_explanation(i).prediction['classes']
        counterfactual_explanations.plot_explanation(i, n_features_to_show=features_to_plot, filename=f"{dataSet}/{dataSet}_counterfactual_Mondrian_i{i}_c{testY[i]}_p{predicted}.png")

    print('Mondrian Counterfactual Explanations done')
