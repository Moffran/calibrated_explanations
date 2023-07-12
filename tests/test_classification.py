from __future__ import absolute_import

import unittest
import pytest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
from lime.discretize import EntropyDiscretizer
# from shap import Explainer

from calibrated_explanations import CalibratedExplainer, BinaryDiscretizer, BinaryEntropyDiscretizer

model = 'RF'
def load_binary_dataset():
    dataSet = 'diabetes_full'
    delimiter = ','
    num_to_test = 4
    target = 'Y'

    fileName = 'data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)

    X, y = df.drop(target,axis=1), df[target] 
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]
    columns = X.columns
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
    # # sort targets to make sure equal presence of both classes in test set (see definition of test_index after outer loop below)
    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx,:], y.values[idx]
    # Select num_to_test/2 from top and num_to_test/2 from bottom of list of instances
    test_index = np.array([*range(int(num_to_test/2)), *range(no_of_instances-1, no_of_instances-int(num_to_test/2)-1,-1)])
    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)   
    trainCalX, testX = X[train_index,:], X[test_index,:]
    trainCalY, testY = y[train_index], y[test_index]
    # trainCalX,trainCalY = shuffle(trainCalX, trainCalY)
    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42, stratify=trainCalY)
    return trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, columns

def load_multiclass_dataset():
    dataSet = 'glass'
    delimiter = ','
    num_to_test = 12
    model_name = 'RF'
    # print(dataSet)

    fileName = 'data/Multiclass/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=delimiter)
    target = 'Type'

    # df.convert_objects()

    df = df.dropna()
    categorical_features = []
    categorical_labels = {}
    for c, col in enumerate(df.columns):
        if df[col].dtype == object:
            df[col] = df[col].str.replace("'", "")
            df[col] = df[col].str.replace('"', '')
            if col != target:
                categorical_features.append(c)
                categorical_labels[c] = dict(zip(range(len(np.unique(df[col]))),np.unique(df[col])))
            else:
                target_labels = dict(zip(range(len(np.unique(df[col]))),np.unique(df[col])))
            mapping = dict(zip(np.unique(df[col]), range(len(np.unique(df[col])))))
            if len(mapping) > 5:
                counts = df[col].value_counts().sort_values(ascending=False)
                id = 0
                for key, count in counts.items():
                    if count > 5:
                        id += 1
                        continue
                    mapping[key] = id
            df[col] = df[col].map(mapping)

    X, y = df.drop(target,axis=1), df[target] 
    columns = X.columns
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
    # # sort targets to make sure equal presence of both classes in test set (see definition of test_index after outer loop below)
    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx,:], y.values[idx]
    test_idx = []
    idx = list(range(no_of_instances))
    for i in range(no_of_classes):
        test_idx.append(np.where(y == i)[0][0:int(num_to_test/no_of_classes)])
    test_index = np.array(test_idx).flatten()
    # Select num_to_test/2 from top and num_to_test/2 from bottom of list of instances
    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)   
    trainCalX, testX = X[train_index,:], X[test_index,:]
    trainCalY, testY = y[train_index], y[test_index]
    # trainCalX,trainCalY = shuffle(trainCalX, trainCalY)
    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42, stratify=trainCalY)
    return trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, columns

def get_classification_model(model_name, trainX, trainY):
    t1 = DecisionTreeClassifier()
    r1 = RandomForestClassifier(n_estimators=100)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] 
    model.fit(trainX,trainY)  
    return model, model_name



class TestCalibratedExplainer(unittest.TestCase):
    def assertExplanation(self, exp):
        for i, instance in enumerate(exp.x):
            boundaries = exp.CE.rule_boundaries(instance)
            for f in range(exp.CE.num_features):
                # assert that instance values are covered by the rule conditions
                assert instance[f] >= boundaries[f][0] and instance[f] <= boundaries[f][1]
        return True

    def test_binary_ce(self):
        trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, feature_names = load_binary_dataset()
        model, model_name = get_classification_model('RF', trainX, trainY)
        cal_exp = CalibratedExplainer(
            model, 
            calX, 
            calY,
            feature_names=feature_names, 
            discretizer='binary', 
            categorical_features=categorical_features, 
            mode='classification',
        )
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, BinaryDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()
        
        cal_exp.set_discretizer('binaryEntropy')
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()    
        
        cal_exp.set_discretizer('entropy')
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, EntropyDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()    

    @unittest.skip('Test passes locally.  Skipping provisionally.')
    def test_multiclass_ce(self):
        trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, feature_names = load_multiclass_dataset()
        model, model_name = get_classification_model('RF', trainX, trainY)
        cal_exp = CalibratedExplainer(
            model, 
            calX, 
            calY,
            feature_names=feature_names, 
            discretizer='binary', 
            categorical_features=categorical_features, 
            mode='classification',
        )
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, BinaryDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()        
        
        cal_exp.set_discretizer('binaryEntropy')
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()    
        
        cal_exp.set_discretizer('entropy')
        exp = cal_exp(testX)
        self.assertIsInstance(exp.CE.discretizer, EntropyDiscretizer)
        self.assertExplanation(exp)
        exp.add_conjunctive_counterfactual_rules()
        exp.get_counterfactual_rules()
        exp.add_conjunctive_factual_rules()
        exp.get_factual_rules()    


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
