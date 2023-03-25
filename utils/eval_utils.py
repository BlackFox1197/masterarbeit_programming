import numpy as np
import pandas as pd
import sklearn.metrics
import seaborn as sns
from matplotlib import pyplot as plt
from numba import float64


def classificationReport(true_codes, pred_codes, sortedLabelStrings, printReport = True, return_string = True, return_acucracy = False):
    assert len(true_codes) == len(pred_codes)
    length = len(sortedLabelStrings)
    ground_true_occurances = np.array([np.count_nonzero(np.array(true_codes) == i) for i in range(len(sortedLabelStrings))])
    correct_occurances = np.sum(np.asarray(pred_codes) == np.asarray(true_codes))

    tp = np.array([np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) == i)) for i in range(length)], dtype=float)
    tn = np.array([np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) != i)) for i in range(length)], dtype=float)
    fp = np.array([np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) != i)) for i in range(length)], dtype=float)
    fn = np.array([np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) == i)) for i in range(length)], dtype=float)




    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
    recall = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
    accuracy = correct_occurances/len(pred_codes)


    labelMaxFormat = '{0: >'+str(len(max(max(sortedLabelStrings, key=len), "avg")))+'}'
    fill = labelMaxFormat.format('')

    string = f"{fill}   accuracy  precision  recall   support\n"
    for i in range(length):
        lableStr = f"{labelMaxFormat.format(sortedLabelStrings[i])}     {accuracy:.3f}     {precision[i]:.3f}     {recall[i]:.3f}    {ground_true_occurances[i]}\n"
        string += lableStr
    string += f"{labelMaxFormat.format('')}                                  {np.sum(ground_true_occurances)}\n"
    string += "\n \n"
    string += f"{labelMaxFormat.format('avg')}     {accuracy:.3f}     {np.mean(precision):.3f}     {np.mean(recall):.3f}    \n"
    if(printReport):
        print(string)

    if(return_acucracy):
        return  accuracy

    if(return_string):
        return string

    return tp, tn, fp, fn


def confustion_matrix_heatmap(true_codes, pred_codes, sortedLabelStrings, title, show_legend = True, normalize = True):
    length = len(sortedLabelStrings)
    #matrix = sklearn.metrics.confusion_matrix(true_codes, pred_codes) / 100
    matrix = sklearn.metrics.confusion_matrix(true_codes, pred_codes)
    if normalize:
        matrix = matrix / np.sum(matrix, axis=1)[:, None]
    df = pd.DataFrame(matrix, index=sortedLabelStrings, columns=sortedLabelStrings)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title, fontsize=21)
    sns.heatmap(df, annot=True, cmap="viridis", cbar=show_legend)
    plt.show()


def confusion_matrix(true_codes, pred_codes, sortedLabelStrings, printReport = True):
    length = len(sortedLabelStrings)
    matrix = sklearn.metrics.confusion_matrix(true_codes, pred_codes)

    labelMaxFormat = '{0: >' + str(max(len(max(sortedLabelStrings, key=len)), 6)+4) + '}'
    fill = labelMaxFormat.format('')

    headerStr = fill
    bodyStr = ""
    for i in range(length):
        label = f"{labelMaxFormat.format(sortedLabelStrings[i])}"
        headerStr += label
        bodyStr += label
        percantages = matrix[i] / max(np.sum(matrix[i]), 1)
        for j in range(length):
            bodyStr += labelMaxFormat.format(f'{percantages[j]:.3f}')
        bodyStr += labelMaxFormat.format(f"{np.sum(matrix[i])}\n")
    headerStr += labelMaxFormat.format("support\n")

    if(printReport):
        print(headerStr + bodyStr)
    return headerStr+bodyStr



#print(classificationReport([2,1,1,1,3], [2,1,0,2,3], ["a", "b", "c", "dsss"]))
#print(confusion_matrix([2,1,1,1,3], [2,1,0,2,3], ["a", "b", "c", "dssssssssss"]))