
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import shuffle
from sklearn.metrics import (brier_score_loss, log_loss, accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from calibrated_explanations.VennAbers import VennAbers
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def bin_total(y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1
    return np.bincount(binids, minlength=len(bins))

def ece(y_prob, fop, mpv, n_bins=10):
    bins = bin_total(y_prob, n_bins)
    bins = bins[bins != 0]
    w = bins / np.sum(bins)
    return np.sum(w * abs(fop - mpv))

def fix_class_missing(proba, class_is_missing, unique_y, unique_train_y):
    if not class_is_missing:
        return proba
    new_proba = np.zeros([proba.shape[0], len(unique_y)])
    for y_i, u_y in enumerate(unique_train_y):
        idx = np.searchsorted(unique_y, u_y)
        new_proba[:, idx] = proba[:, y_i]
    return new_proba

# MAIN PROGRAM
outerloop = 10
eval_matrix = []

for dataSet in ["iris", "tae", "image", "wineW","wineR", "wine", "glass", "vehicle", "cmc", "balance", "wave", "vowel", "cars", "steel", "heat", "cool", "user", "whole", "yeast" ]:

# for dataSet in ["ecoli"]:

    fileName="data/Multiclass/multi/"+dataSet+".csv"
    df = pd.read_csv(fileName, sep=';')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    X,y = shuffle(X, y)
    if min(y) == 1:
        y = y - 1

    no_of_classes = len(np.unique(y))

    # m1 = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.01)
    # m2 = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.01)
    m1 = RandomForestClassifier()
    m2 = RandomForestClassifier()

    iso = IsotonicRegression(out_of_bounds="clip")

    yall = np.array([])
    uncal_probs_all = np.empty(shape=[0, no_of_classes])
    va_probs_all = np.empty(shape=[0, no_of_classes])
    va_l_probs_all = np.array([])
    va_m_probs_all = np.array([])
    uncal_preds_all = np.array([])
    va_preds_all = np.array([])
    low = np.array([])
    high = np.array([])
    treeSize_noncal = 0
    treeSize_va = 0

    for x in range(outerloop):
        uncal_probs = np.ones((len(y), no_of_classes), dtype=np.float32)
        uncal_preds = np.zeros(len(y))
        va_preds = np.zeros(len(y))
        l = np.ones((len(y), no_of_classes), dtype=np.float32)
        h = np.ones((len(y), no_of_classes), dtype=np.float32)
        vap = np.zeros(len(y))
        va_probs = np.ones((len(y), no_of_classes), dtype=np.float32)

        kf = StratifiedKFold(n_splits=10)

        for train_index, test_index in kf.split(X, y):
            trainCalX, testX = X[train_index], X[test_index]
            trainCalY, testY = y[train_index], y[test_index]
            m1.fit(trainCalX, trainCalY)
            # treeSize_noncal = treeSize_noncal + m1.tree_.node_count

            trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33)

            uniqueY = np.unique(y)
            uniqueTrainY = np.unique(trainCalY)
            class_missing = len(uniqueTrainY) != len(uniqueY)

            s = len(trainY)
            

            m2.fit(trainX, trainY)
            # treeSize_va = treeSize_va + m2.tree_.node_count

            uncal_preds[test_index] = m1.predict(testX)
            uncal_probs[test_index, :] = fix_class_missing(m1.predict_proba(testX), class_missing, uniqueY, uniqueTrainY)
            

            uniqueTrainY = np.unique(trainY)
            class_missing = len(uniqueTrainY) != len(uniqueY)

            cal_preds = m2.predict(calX)
            cal_probs = fix_class_missing(m2.predict_proba(calX), class_missing, uniqueY, uniqueTrainY)

            test_preds = m2.predict(testX)
            test_probs = fix_class_missing(m2.predict_proba(testX), class_missing, uniqueY, uniqueTrainY)

            va = VennAbers(cal_probs, calY, m2)
            tmp = va.predict_proba(testX, output_interval=True)
            va_probs[test_index, :] = tmp[0]
            va_preds[test_index] = np.argmax(tmp[0], axis=1)
            l[test_index, :] = tmp[1]
            h[test_index, :] = tmp[2]

            

        yall = np.append(yall, y)
        uncal_probs_all = np.append(uncal_probs_all, uncal_probs, axis=0)
      
        va_probs_all = np.append(va_probs_all, va_probs, axis=0)
        uncal_preds_all = np.append(uncal_preds_all, uncal_preds)
       
        va_preds_all = np.append(va_preds_all, va_preds)
        low = np.vstack([low, l]) if low.size else l
        high = np.vstack([high, h]) if high.size else h

    uncal_probs_predicted = np.amax(uncal_probs_all, 1)
    uncal_corrects = (uncal_preds_all == yall).astype(int)
    uncal_diff = np.mean(uncal_probs_predicted) - accuracy_score(yall, uncal_preds_all)

    va_probs_predicted = np.amax(va_probs_all, 1)
    va_corrects = (va_preds_all == yall).astype(int)
    va_diff = np.mean(va_probs_predicted) - accuracy_score(yall, va_preds_all)

    fop_uncal, mpv_uncal = calibration_curve(uncal_corrects, uncal_probs_predicted, n_bins=10)
    fop_va, mpv_va = calibration_curve(va_corrects, va_probs_predicted, n_bins=10)

    print(dataSet)

    plt.figure(figsize=(7, 7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name in ['Uncal','VA']:#
        if name == 'Uncal':
            y_test = uncal_corrects
            prob_pos = uncal_probs_predicted
            ec = "%.3f" % (ece(uncal_probs_predicted, fop_uncal, mpv_uncal))
        if name == 'VA':
            y_test = va_corrects
            prob_pos = va_probs_predicted
            ec = "%.3f" % (ece(va_probs_predicted, fop_va, mpv_va))

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        line_new = name + ": ECE=" + ec
        # line_new = f"{name:<12}  {ec:<12}"
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s" % (line_new,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                    histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Reliability')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", ncol=2)

    plt.tight_layout()
    plt.savefig('debug/multiclass/plots/' + dataSet + '_reliability.png')
    # plt.show()
    plt.close()


    va_low_predicted = [low[i,j] for i,j in enumerate(np.argmax(va_probs_all, axis=1))]
    va_high_predicted = [high[i,j] for i,j in enumerate(np.argmax(va_probs_all, axis=1))]
    va_corrects = (va_preds_all == yall).astype(int)
    va_diff = np.mean(va_probs_predicted) - accuracy_score(yall, va_preds_all)

    fop_low, mpv_low = calibration_curve(va_corrects, va_low_predicted, n_bins=10)
    fop_high, mpv_high = calibration_curve(va_corrects, va_high_predicted, n_bins=10)

    plt.figure(figsize=(7, 7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name in ['Uncal', 'VA', 'VA Low', 'VA High']:
        if name == 'Uncal':
            y_test = uncal_corrects
            prob_pos = uncal_probs_predicted
            ec = "%.3f" % (ece(uncal_probs_predicted, fop_uncal, mpv_uncal))
            line_new = name + ": ECE=" + ec

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                        histtype="step", lw=2)
            
        if name == 'VA':
            y_test = va_corrects
            prob_pos = va_probs_predicted
            ec = "%.3f" % (ece(va_probs_predicted, fop_va, mpv_va))
            line_new = name + ": ECE=" + ec

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                        histtype="step", lw=2)

        if name == 'VA Low':
            y_test = va_corrects
            prob_pos = va_low_predicted
            ec = "%.3f" % (ece(va_low_predicted, fop_low, mpv_low))
            line_new = name
        if name == 'VA High':
            y_test = va_corrects
            prob_pos = va_high_predicted
            ec = "%.3f" % (ece(va_high_predicted, fop_high, mpv_high))
            line_new = name

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        # line_new = f"{name:<12}  {ec:<12}"
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s" % (line_new,))
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(dataSet)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", ncol=2)

    plt.tight_layout()
    plt.savefig('debug/multiclass/plots/' + dataSet + '.png')
    # plt.show()
    plt.close()

    eval_matrix.append([dataSet, 'Brier', 'NoCal', brier_score_loss(uncal_corrects, uncal_probs_predicted)])
    eval_matrix.append([dataSet, 'Acc', 'NoCal', accuracy_score(yall, uncal_preds_all)])
    eval_matrix.append([dataSet, 'Log', 'NoCal', log_loss(uncal_corrects, uncal_probs_predicted)])
    eval_matrix.append([dataSet, 'Diff', 'NoCal', uncal_diff])
    eval_matrix.append([dataSet, 'ECE', 'NoCal', ece(uncal_probs_predicted, fop_uncal, mpv_uncal)])
    eval_matrix.append([dataSet, 'treeSize', 'NoCal', (treeSize_noncal / (10 * outerloop))])
    eval_matrix.append([dataSet, 'AUC', 'NoCal', roc_auc_score(uncal_corrects, uncal_probs_predicted)])
    eval_matrix.append([dataSet, 'Brier', 'VA', brier_score_loss(va_corrects, va_probs_predicted)])
    eval_matrix.append([dataSet, 'Acc', 'VA', accuracy_score(yall, va_preds_all)])
    eval_matrix.append([dataSet, 'Log', 'VA', log_loss(va_corrects, va_probs_predicted)])
    eval_matrix.append([dataSet, 'Diff', 'VA', va_diff])
    eval_matrix.append([dataSet, 'ECE', 'VA', ece(va_probs_predicted, fop_va, mpv_va)])
    eval_matrix.append([dataSet, 'treeSize', 'VA', (treeSize_va / (10 * outerloop))])
    eval_matrix.append([dataSet, 'AUC', 'VA', roc_auc_score(va_corrects, va_probs_predicted)])
    eval_matrix.append([dataSet, 'Low', 'VA', np.mean([low[i,j] for i,j in enumerate(np.argmax(va_probs_all, axis=1))])])
    eval_matrix.append([dataSet, 'High', 'VA', np.mean([high[i,j] for i,j in enumerate(np.argmax(va_probs_all, axis=1))])])

    evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Metric', 'Criteria', 'Value'])
    evaluation_matrix.to_csv(r'debug/multiclass/results_COPA_2024.csv', index=True, header=True, sep=';')

