"""

"""

from experiment_utils import (VennAbers, ece, lime_fidelity, shap_fidelity, clip)
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
from lime import lime_tabular
from shap import Explainer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import time
import pickle
def debug_print(message, debug=True):
    if debug:
        print(message)

outerloop = 1 
k=10 
number_of_bins = 10
eval_matrix = []
is_debug = True

descriptors = ['uncal','platt','va',] 
Descriptors = {'uncal':'Uncal','platt': 'Platt','va': 'VA'}
models = ['xGB',] 
explainers = ['ce','lime','shap',] #

datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1",}
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]#7,21
tic_all = time.time()
for dataset in klara:
    dataSet = datasets[dataset]

    tic_data = time.time()
    print(dataSet)
    fileName = '../Data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=';')
    Xn, y = df.drop('Y',axis=1), df['Y'] 

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15) 
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True)
    s2 = SVC(probability=True)
    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    h1 = HistGradientBoostingClassifier()
    h2 = HistGradientBoostingClassifier()
    g1 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    g2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    
    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn),'SVM': (s1,s2,"SVM",Xn),'DT': (t1,t2,"DT",Xn),'HGB': (h1,h2,"HGB",Xn)}
    model_struct = [model_dict[model] for model in models]

    has_pickle = False
    for c1, c2, alg, X in model_struct:
        try:
            all_results = pickle.load(open('../pickle/' + dataSet +' '+ alg + "- fidelity5.pkl", 'rb'))
            results = all_results['results']
            explanations = all_results['explanations']
            print('results unpickled')
            has_pickle = True
        except FileNotFoundError:
            has_pickle = False

        if not(has_pickle):
            tic_algorithm = time.time()
            debug_print(dataSet+' '+alg)
            results = {}
            results['calibrators']=[]
            results['yall']=np.array([])
            results['low']=np.array([])
            results['high']=np.array([])
            for desc in descriptors:
                results[desc] = {}
                for explain in explainers:
                    results[desc][explain] = {}
                    for metric in ['preds',]:
                        results[desc][explain][metric] = np.array([])
                    results[desc][explain]['proba'] = np.empty(shape=[0,no_of_classes])


            for x in range(outerloop):
                local = {}
                for metric in ['low','high']:
                    local[metric] = np.zeros(len(y))
                for desc in descriptors:
                    local[desc] = {}
                    for explain in explainers:    
                        local[desc][explain] = {}
                        local[desc][explain]['proba'] =  np.ones((len(y),no_of_classes), dtype=np.float32 ) 
                        for metric in ['preds',]:
                            local[desc][explain][metric] = np.zeros(len(y))
                        
                explanations = {}  
                for desc in descriptors:              
                    explanations[desc] = {}
                    for explain in explainers:                    
                        explanations[desc][explain] = {}
                        for metric in ['explanation']:
                            explanations[desc][explain][metric] = {}
                        for metric in ['abs_rank','values','fidelity','proba_exp','avg_prob1','weight']:
                            explanations[desc][explain][metric] = np.zeros((len(y),no_of_features))

                debug = {}
                kf = StratifiedKFold(n_splits=k)
                kn = 0

            
                for train_index, test_index in kf.split(X,y):  
                    calibrators = {} 
                    for desc in descriptors:
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

                    explainer = lime_tabular.LimeTabularExplainer(
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
                        calibrators['va']['model'] = VennAbers(calX, calY, c2)
                        for explain in explainers:                                    
                            local['va'][explain]['proba'][test_index,:] = calibrators['va']['model'].predict_proba(testX)
                            local['va'][explain]['preds'][test_index] = calibrators['va']['model'].predict(testX)
                    calibrators['data'] = {'trainX':trainX,'trainY':trainY,'calX':calX,'calY':calY,'testX':testX,'testY':testY,'test_index':test_index,}

                    debug_print(str(kn)  + ': ' + dataSet + ' ' + alg + ' ' + desc , is_debug)
                    np.random.seed(1337)
                    for explain in explainers:
                        for desc in descriptors:
                            calibrator = calibrators[desc]['model']
                            
                            if explain == 'shap':
                                predict_fn = lambda x:calibrator.predict_proba(x)[:,1]

                                explainer = Explainer(predict_fn, calX)
                                explanation = explainer(testX)          
                                explanations[desc][explain]['explanation'][str(kn)] = explanation
                                # Sort features and values to get identical structures from lime and shap for easier evaluation
                                explanations[desc][explain]['abs_rank'][test_index,:] = np.flip(np.argsort(np.abs(explanation.values), axis=1), axis=1)
                                explanations[desc][explain]['values'][test_index,:] = np.array([[explanation.values[i, int(j)] for j in explanations[desc][explain]['abs_rank'][ii]] for i, ii in enumerate(test_index)])  
                                calibrators[desc][explain].append(explanation) 
                            

                            if explain == 'lime':
                                predict_fn = lambda x:calibrator.predict_proba(x)
                                        
                                res_struct = {}
                                for m in ['explanation', 'fidelity', 'proba_exp', 'avg_prob1', 'weight','abs_rank','values']:
                                    res_struct[m] = [] 
                                explainer = lime_tabular.LimeTabularExplainer(
                                                    training_data=np.array(calX),
                                                    # feature_names=calX.columns,
                                                    class_names=np.unique(calY),
                                                    mode='classification', 
                                                    random_state=1337, 
                                                )
                                
                                for i, j in enumerate(test_index):
                                    x = testX[i]
                                    exp = explainer.explain_instance(x, predict_fn = predict_fn, num_features=len(x))
                                    fidelity, tmp = lime_fidelity(exp, explainer, calibrator, x)
                                    res_struct['fidelity'].append(list(fidelity)) 
                                    res_struct['proba_exp'].append(tmp['pw']) 
                                    res_struct['avg_prob1'].append(tmp['p_one']) 
                                    res_struct['weight'].append(tmp['weight']) 
                                    res_struct['explanation'].append(exp) 
                                    # Sort features and values to get identical structures from lime and shap for easier evaluation
                                    res_struct['abs_rank'].append([res_struct['explanation'][i].local_exp[1][j][0] for j in range(len(x))])                                
                                    res_struct['values'].append([res_struct['explanation'][i].local_exp[1][j][1] for j in range(len(x))])    
                                    explanations[desc][explain]['explanation'][str(j)] = res_struct['explanation'][i]

                                for metric in ['abs_rank','values','fidelity','proba_exp','avg_prob1','weight']:
                                    explanations[desc][explain][metric][test_index,:] = res_struct[metric] 
                                calibrators[desc][explain].append(res_struct) 
                    
                    results['calibrators'].append(calibrators)

                for explain in explainers:
                    for desc in descriptors:
                        for metric in ['preds', 'proba']:
                            results[desc][explain][metric]=np.append(results[desc][explain][metric], local[desc][explain][metric],axis=0)
                            
                for metric in ['low','high']:
                    results[metric]=np.append(results[metric],local[metric])
                results['yall']=np.append(results['yall'],y) 


            all_results = {'explanations':explanations, 'results':results}
            a_file = open('../pickle/' + dataSet +' '+  alg + "- experiment_1_1.pkl", "wb")
            pickle.dump(all_results, a_file)
            a_file. close() 
        for explain in explainers:

            # Evaluation lime/shap
            for calib in descriptors:                              
                idx = 0
                fop, mpv = calibration_curve(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)), n_bins=number_of_bins)
                eval_matrix.append([dataSet, alg,'Brier', explain, '', '', calib, brier_score_loss(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'Log', explain, '', '', calib, log_loss(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'ECE', explain, '', '', calib, ece(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)), fop, mpv)])
                fop, mpv = calibration_curve(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]), n_bins=number_of_bins)
                eval_matrix.append([dataSet, alg,'Brier', explain, '', 1, calib, brier_score_loss(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])
                eval_matrix.append([dataSet, alg,'Log', explain, '', 1, calib, log_loss(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])
                eval_matrix.append([dataSet, alg,'ECE', explain, '', 1, calib, ece(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]), fop, mpv)])
                eval_matrix.append([dataSet, alg,'MSE', explain, 'compare', '', calib, mean_squared_error(results[calib][explain]['proba'][:,1], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'MAE', explain, 'compare', '', calib, mean_absolute_error(results[calib][explain]['proba'][:,1], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'MSE', explain, 'compare', 1, calib, mean_squared_error(results[calib][explain]['proba'][:,1], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])
                eval_matrix.append([dataSet, alg,'MAE', explain, 'compare', 1, calib, mean_absolute_error(results[calib][explain]['proba'][:,1], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])

                # eval_matrix.append([dataSet, alg, 'fidelity_score',explain, '', '', calib,np.mean([explanations[calib][explain]['explanation'][str(i)].score for i in range(no_of_instances)])]) #
                eval_matrix.append([dataSet, alg, 'fidelity',explain, '', 1, calib,np.mean(explanations[calib][explain]['fidelity'])]) #
                eval_matrix.append([dataSet, alg, 'ma_fidelity',explain, '', 1, calib,np.mean(np.abs(explanations[calib][explain]['fidelity']))]) #
                
        evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Algorithm', 'Metric', 'Explainer', 'Type', 'NumFeatures', 'Comparison','Value'])
        evaluation_matrix.to_csv('..\results\experiment_1_1.csv', index=True, header=True, sep=';')
        toc_algorithm = time.time()
        debug_print(dataSet + "-" + alg,is_debug )


    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  


toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 