import numpy as np
from scipy.stats import spearmanr
import collections

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

def find_bin(rules, rule):
    return rules.index(rule)

def lime_fidelity(exp, explainer, model, instance, feature=None):
    if feature is None:
        fidelity = np.zeros(len(instance))
        res = {'p_one':[], 'weight':[], 'pw':[]}
        for i in range(len(instance)):
            fidelity[i], tmp = lime_fidelity(exp, explainer, model, instance.copy(), i)
            for m in ['p_one','weight','pw']:
                res[m].append(tmp[m])
        return fidelity, res
    feature_idx = exp.local_exp[1][feature][0]
    bin = find_bin(explainer.discretizer.names[feature_idx], exp.as_list()[feature][0])
    pred = exp.predict_proba[1]
    normalized_frequencies = explainer.feature_frequencies[feature_idx].copy()
    if len(normalized_frequencies) > 3:
        normalized_frequencies[bin] = 0
        normalized_frequencies /= normalized_frequencies.sum()
    elif len(normalized_frequencies) > 1:
        if bin == len(normalized_frequencies):
            bin -= 1 # if the rule is X > Y when fewer than four bins, then it always correspond to the last of the bins.
        normalized_frequencies[bin] = 0
        normalized_frequencies /= normalized_frequencies.sum()

    average_pone=0
    weight = exp.local_exp[1][feature][1]
    for j in range(len(normalized_frequencies)):
        instance[feature_idx]=explainer.discretizer.means[feature_idx][j]
        if isinstance(model,collections.Callable):
            p_one = model(instance.reshape(1, -1))[0]
        else:
            p_one = model.predict_proba(instance.reshape(1, -1))[0,1]
        average_pone += p_one * normalized_frequencies[j]
    return 1 - (pred - average_pone - weight), {'p_one':average_pone, 'weight':weight, 'pw':average_pone + weight}



def shap_fidelity(explanation, explainer, model, instances, trainX=None):
    no_features = len(instances[0])
    no_instances = len(instances[:,0])
    average_pone = np.zeros((no_instances, no_features))
    fidelity = np.zeros((no_instances, no_features))
    proba_exp = np.zeros((no_instances, no_features))
    weight = explanation[0].values
    pred = model.predict_proba(instances)[:,1]
    average_pone = np.array([[pred[i] for j in range(no_features)] for i in range(no_instances)] )

    assert not np.any(np.isnan(average_pone)),'finns nan'
    from shap.utils import MaskedModel
    from shap.explainers import Exact, Permutation
    if trainX is not None:
        for i in range(no_features):
            p_one = 0 
            values = np.unique(trainX[:,i])
            for n in range(no_instances):
                val = values[values != instances[n,i]]
                if len(val) > 0:
                    instance = instances[n,:].copy() 
                    n_instances = np.array([instance for j in range(len(val))])
                    n_instances[:,i] = val
                    average_pone[n,i] = np.average(model.predict_proba(n_instances)[:,1])
    else:
        for n in range(no_instances):
            # print('.')
            instance = instances[n,:].copy()
            fm = MaskedModel(explainer.model, explainer.masker, explainer.link, instances[n,:])

            if issubclass(type(explainer), Permutation):
                max_evals = 3 * 2 * len(fm)
                # loop over many permutations
                inds = fm.varying_inputs()
                inds_mask = np.zeros(len(fm), dtype=bool)
                inds_mask[inds] = True
                masks = np.zeros(2*len(inds)+1, dtype=int)
                masks[0] = MaskedModel.delta_mask_noop_value
                npermutations = 1#max_evals // (2*len(inds)+1)
                outputs = []
                changed = None
                if len(inds) > 0:
                    p_one = np.zeros(no_features)
                    count = np.zeros(no_features)
                    for _ in range(npermutations):
                        np.random.shuffle(inds)

                        # create a large batch of masks to evaluate
                        i = 1
                        for ind in inds:
                            masks[i] = ind
                            i += 1
                        for ind in inds:
                            masks[i] = ind
                            i += 1
                        masked_inputs, varying_rows = explainer.masker(masks, *fm.args)

                        subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
                        subset_masked_inputs = subset_masked_inputs[0]#[np.random.rand(varying_rows.sum()) < 0.3,:]
                        # evaluate the masked model 
                        for i in inds:
                            instance = instances[n,:].copy()  
                            subtract = 0  
                            for v in np.unique(subset_masked_inputs[:,i]):
                                if v == instances[n,i]:
                                    subtract = 1
                                    continue
                                instance[i] = v
                                p_one[i] += model.predict_proba([instance])[:,1]
                            count[i] += len(np.unique(subset_masked_inputs[:,i])) - subtract
                    for i in inds:
                        average_pone[n,i] = p_one[i]/count[i]

            elif issubclass(type(explainer), Exact):

                inds = fm.varying_inputs()        
                delta_indexes = explainer._cached_gray_codes(len(inds))
                extended_delta_indexes = np.zeros(2**len(inds), dtype=int)
                for i in range(2**len(inds)):
                    if delta_indexes[i] == MaskedModel.delta_mask_noop_value:
                        extended_delta_indexes[i] = delta_indexes[i]
                    else:
                        extended_delta_indexes[i] = inds[delta_indexes[i]]

                # run the model
                masked_inputs, varying_rows = explainer.masker(extended_delta_indexes, *fm.args)
                num_varying_rows = varying_rows.sum(1)

                subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
                subset_masked_inputs = subset_masked_inputs[0]

                for i in inds:
                    p_one = 0
                    instance = instances[n,:].copy()  
                    subtract = 0  
                    for v in np.unique(subset_masked_inputs[:,i]):
                        if v == instances[n,i]:
                            subtract = 1
                            continue
                        instance[i] = v
                        p_one += model.predict_proba([instance])[:,1]
                    average_pone[n,i] = p_one/(len(np.unique(subset_masked_inputs[:,i])) - subtract)

    for feature in range(no_features):
        proba_exp[:,feature] = average_pone[:,feature] - weight[:,feature]
        fidelity[:,feature] = 1 - (pred - proba_exp[:,feature])

    return fidelity, average_pone, weight, proba_exp

def debug_print(message, debug=True):
    if debug:
        print(message)



def topRanked(series1, series2, metric='difference', order_importance='decreasing'):
    """
    Compute the difference between two series of ranked feauters

    Returns the difference between two series of ranked features.
    Positions (can) have decreasing importance, with the highest ranked feature being most important.
    
    Parameters
    ----------
    series1 : array-like
        matrix of ranked features per instance
    series2 : array-like
        matrix of ranked features per instance
    metric : 'difference' (deafult), 'spearman', optional
        'difference' calculates the proprtion of ranks that are the same and 'spearman' calculates the spearman correlation
    order_importance : 'decreasing' (deafult), 'identical', optional
        'decreasing' values the first position with 1, the second with 1/2 etc, whereas 'identical' does not consider rank position 
    """    
    
    assert(series1.size == series2.size)
    if len(series1.shape) == 1:
        if metric == 'difference':
            return np.mean([int(item) for item in series1 == series2])
        elif metric == 'spearman':
            return spearmanr(series1, series2).correlation
    m = 1
    m_sum = 0
    result = 0
    for col in range(series1.shape[1]):    
        m_sum += m   
        if metric == 'difference':
            result += m * np.mean([int(item) for item in series1[:,col] == series2[:,col]])
        elif metric == 'spearman':
            result += m * spearmanr(series1[:,col], series2[:,col]).correlation
        if order_importance == 'decreasing':
            m = m/2 

    return result/m_sum