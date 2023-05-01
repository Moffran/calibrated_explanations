"""

"""

from experiment_utils import (ece, lime_fidelity, shap_fidelity, clip)
import VennAbers
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from lime.lime_tabular import LimeTabularExplainer
from shap import Explainer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from calibrated_explanations import CalibratedExplainer
import copy
import time
import pickle
from tqdm import tqdm


# -------------------------------------------------------

def debug_print(message, debug=True):
    if debug:
        print(message)
        
# ------------------------------------------------------

num_repetitions = 50
num_to_test=10 
# number_of_bins = 10
eval_matrix = []
is_debug = True

descriptors = ['uncal'] #['uncal','platt','va',]'uncal',
Descriptors = {'uncal':'Uncal','va': 'VA'} # 'platt': 'Platt',
models = ['xGB'] # ['xGB','RF','DT','SVM',] # 'NN','RF',
explainers = ['ce'] # ,'lime','shap' 

datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1"} 
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
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
    categorical_features = [i for i in range(no_of_features) if len(np.unique(Xn.iloc[:,i])) < 10]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15) 
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True) # Skala även input
    s2 = SVC(probability=True)
    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    h1 = HistGradientBoostingClassifier(categorical_features=categorical_features)
    h2 = HistGradientBoostingClassifier(categorical_features=categorical_features)
    g1 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False, categorical_features=categorical_features)
    g2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False, categorical_features=categorical_features)
    
    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn),'SVM': (s1,s2,"SVM",Xn),'DT': (t1,t2,"DT",Xn),'HGB': (h1,h2,"HGB",Xn)}
    model_struct = [model_dict[model] for model in models]

    for c1, c2, alg, X in model_struct:       
        tic_algorithm = time.time()
        debug_print(dataSet+' '+alg)
        results = {}
        results['calibrators']=[]
        results['yall']=np.array([])
        results['low']=np.array([])
        results['high']=np.array([])
        for desc in ['uncal','va']:
            results[desc] = {}
            for explain in explainers:
                results[desc][explain] = {}
                for metric in ['preds',]:
                    results[desc][explain][metric] = np.array([])
                results[desc][explain]['proba'] = np.empty(shape=[0,no_of_classes])


        local = {}
        for metric in ['low','high']:
            local[metric] = np.zeros(len(y))
        for desc in ['uncal','va']:
            local[desc] = {}
            for explain in explainers:    
                local[desc][explain] = {}
                local[desc][explain]['proba'] =  np.ones((len(y),no_of_classes), dtype=np.float32 ) 
                for metric in ['preds',]:
                    local[desc][explain][metric] = np.zeros(len(y))
                
        explanations = {}  
        for desc in ['uncal','va']:              
            explanations[desc] = {}
            for explain in explainers:                    
                explanations[desc][explain] = {}
                for metric in ['explanation']:
                    explanations[desc][explain][metric] = {}
                for metric in ['stability','robustness_rule','robustness_local','robustness_global','ce_time','lime_time','shap_time']:
                    explanations[desc][explain][metric] = []

        kn = 0    
        
        train_index, test_index = train_test_split(range(no_of_instances), test_size=num_to_test,random_state=42)
        calibrators = {} 
        for desc in ['uncal','va']:
            calibrators[desc] = {}
            calibrators[desc]['lime'] = []
            calibrators[desc]['shap'] = []
        kn += 1
        trainCalX, testX = X.iloc[train_index].values, X.iloc[test_index].values
        trainCalY, testY = y.iloc[train_index].values, y.iloc[test_index].values
        trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
        
        if any([alg == cmp for cmp in ['NN','SVM']]):
            sc = StandardScaler()
            trainX = sc.fit_transform(trainX)
            calX = sc.transform(calX)
            testX = sc.transform(testX)
        c2.fit(trainX,trainY)

        explainer = LimeTabularExplainer(
            training_data=np.array(trainX),
            feature_names=X.columns,
            class_names=np.unique(y),
            mode='classification', 
            random_state=42
        )
        calibrators['uncal']['model'] = c2
        for explain in explainers:                  
            local['uncal'][explain]['proba'][test_index,:] = calibrators['uncal']['model'].predict_proba(testX)
            local['uncal'][explain]['preds'][test_index] = calibrators['uncal']['model'].predict(testX)
        if descriptors.__contains__('platt'):
            calibrators['platt']['model'] = CalibratedClassifierCV(base_estimator = c2, cv="prefit")
            calibrators['platt']['model'].fit(calX,calY)
            for explain in explainers:                 
                local['platt'][explain]['proba'][test_index,:] = calibrators['platt']['model'].predict_proba(testX)
                local['platt'][explain]['preds'][test_index] = np.round(calibrators['platt']['model'].predict(testX))
        if descriptors.__contains__('va'):                        
            calibrators['va']['model'] = VennAbers.VennAbers(calX, calY, c2)
            for explain in explainers:                                    
                local['va'][explain]['proba'][test_index,:] = calibrators['va']['model'].predict_proba(testX)
                local['va'][explain]['preds'][test_index] = calibrators['va']['model'].predict(testX)
        calibrators['data'] = {'trainX':trainX,'trainY':trainY,'calX':calX,'calY':calY,'testX':testX,'testY':testY,'test_index':test_index,}

        debug_print(str(kn)  + ': ' + dataSet + ' ' + alg + ' ' + desc , is_debug)
        np.random.seed(1337)
        explain = 'ce'
        for desc in descriptors:
            calibrator = calibrators[desc]['model']                
            
            ce = CalibratedExplainer(calibrator, copy.deepcopy(calX), copy.deepcopy(calY), \
                feature_names=df.columns, \
                categorical_features=categorical_features)

            discretizer = 'binaryEntropy'
            ce.set_discretizer(discretizer)          
            tic = time.time()
            exp = ce(copy.deepcopy(testX))
            toc = time.time()
            explanations[desc][explain]['ce_time'].append(toc-tic)
            print(f"CE time: {np.nanmin(explanations[desc][explain]['ce_time']):.2f}")
            
            predict_fn = lambda x:calibrator.predict_proba(x)[:,1]
            shap = Explainer(predict_fn, calX)
            tic = time.time()
            explanation = shap(testX) 
            toc = time.time()
            explanations[desc][explain]['shap_time'].append(toc-tic) 
            print(f"SHAP time: {np.nanmin(explanations[desc][explain]['shap_time']):.2f}")

            predict_fn = lambda x:calibrator.predict_proba(x) 
            explainer = LimeTabularExplainer(
                                training_data=np.array(calX),
                                class_names=np.unique(calY),
                                mode='classification', 
                                random_state=1337, 
                            )    #      
            tic = time.time()  
            for i, j in enumerate(test_index):
                x = testX[i]
                exp = explainer.explain_instance(x, predict_fn = predict_fn, num_features=len(x))
            toc = time.time()
            explanations[desc][explain]['lime_time'].append(toc-tic) 
            print(f"LIME time: {np.nanmin(explanations[desc][explain]['lime_time']):.2f}")
                
        results['calibrators'].append(calibrators)

        for explain in explainers:
            for desc in descriptors:
                for metric in ['preds', 'proba']:
                    results[desc][explain][metric]=np.append(results[desc][explain][metric], local[desc][explain][metric],axis=0)
                    
        for metric in ['low','high']:
            results[metric]=np.append(results[metric],local[metric])
        results['yall']=np.append(results['yall'],y) 

        all_results = {'explanations':explanations, 'results':results}
        a_file = open('pickle/robustness 2 - ' + dataSet +' '+  alg + ".pkl", "wb")
        pickle.dump(all_results, a_file)
        a_file. close()
        for explain in explainers:
            # Evaluation lime/shap
            for calib in descriptors:                              
                idx = 0
                eval_matrix.append([dataSet, alg,'Robustness_local', explain, '', '', calib, np.nanmean(explanations[desc][explain]['robustness_local'])])
                eval_matrix.append([dataSet, alg,'robustness_global', explain, '', '', calib, np.nanmean(explanations[desc][explain]['robustness_global'])])
                
        evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Algorithm', 'Metric', 'Explainer', 'Type', 'NumFeatures', 'Comparison','Value'])
        evaluation_matrix.to_csv('results\experiment_1_robsutness 2.csv', index=True, header=True, sep=';')
        toc_algorithm = time.time()
        debug_print(dataSet + "-" + alg,is_debug )


    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 