import numpy as np


def classificationReport(true_codes, pred_codes, sortedLabelStrings):
    assert len(true_codes) == len(pred_codes)
    length = len(sortedLabelStrings)
    true_occurances = np.array([np.count_nonzero(np.array(true_codes) == i) for i in range(len(sortedLabelStrings))])
    #correct = np.zeros(length)
    tp = np.zeros(length)
    fp = np.zeros(length)
    fn = np.zeros(length)
    tn = np.zeros(length)
    # for i in range(len(true_codes)):
    #     if(true_codes[i] == pred_codes[i]):
    #         tp[true_codes[i]] += 1
    #         tn += np.ones_like(tp)
    #         tn[true_codes[i]] -= 1
    #     else:
    #         tn += np.ones_like(tp)
    #         tn[pred_codes[i]] -= 1
    #         tn[true_codes[i]] -= 1
    #         fp[pred_codes[i]] += 1
    #         fn[true_codes[i]] += 1

    tp1 = [np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) == i)) for i in range(length)]
    tn1 = [np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) != i)) for i in range(length)]
    fp1 = [np.sum(np.logical_and(np.asarray(pred_codes) == i, np.asarray(true_codes) != i)) for i in range(length)]
    fn1 = [np.sum(np.logical_and(np.asarray(pred_codes) != i, np.asarray(true_codes) == i)) for i in range(length)]









    print(f"precision: {np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)!=0)}")
    print(f"recall: {np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)}")
    print(f"correct classified: {np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)}")
    #print(f"recall: {np.divide(tp, tp+fn, out=np.ones_like(tp), where=(tp+fn)!=0)}")
    return tn



print(classificationReport([2,1,1,1,3], [2,1,0,2,3], ["a", "b", "c", "d"]))