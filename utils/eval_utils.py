import numpy as np
from numba import float64


def classificationReport(true_codes, pred_codes, sortedLabelStrings):
    assert len(true_codes) == len(pred_codes)
    length = len(sortedLabelStrings)
    #true_occurances = np.array([np.count_nonzero(np.array(true_codes) == i) for i in range(len(sortedLabelStrings))])

    tp = np.array([np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) == i)) for i in range(length)], dtype=float)
    tn = np.array([np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) != i)) for i in range(length)], dtype=float)
    fp = np.array([np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) != i)) for i in range(length)], dtype=float)
    fn = np.array([np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) == i)) for i in range(length)], dtype=float)




    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
    recall = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
    accuracy = np.divide(tp+tn, tp+tn+fp+fn)
    print(len(max(sortedLabelStrings, key=len)))

    labelMaxFormat = '{0: >'+str(len(max(sortedLabelStrings, key=len)))+'}'
    fill = labelMaxFormat.format('')

    string = f"{fill}   accuracy  precision  recall \n"
    for i in range(length):
        lableStr = f"{labelMaxFormat.format(sortedLabelStrings[i])}     {accuracy[i]:.3f}     {precision[i]:.3f}     {recall[i]:.3f} \n"
        string += lableStr
    print(string)
    # print(f"precision: {np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)!=0)}")
    # print(f"recall: {np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)}")
    # print(f"accuracy: {np.divide(tp+tn, tp+tn+fp+fn)}")

    #print(f"recall: {np.divide(tp, tp+fn, out=np.ones_like(tp), where=(tp+fn)!=0)}")
    return tp, tn, fp, fn



#print(classificationReport([2,1,1,1,3], [2,1,0,2,3], ["a", "b", "c", "dsss"]))