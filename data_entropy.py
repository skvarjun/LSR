# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import log2
from numpy import linalg,mean,dot
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['text.usetex'] = True

def entropy(X,classes):
    if len(X)<=1:
        return 0
    no_class1=len(np.where(X[:,1]==classes[0])[0])
    no_class2=len(np.where(X[:,1]==classes[1])[0])
    if no_class1==0 or no_class2==0:
        return 0
    p1=no_class1/(no_class1+no_class2)
    p2=1-p1
    return -(p1 * log2(p1)+p2*log2(p2))


def supervised_discretization(D, classes, feature_entropy_threshold, feature_no_bin_threshold):
    range_results = []

    def recursive_supervised_discretization(D):
        if len(D) <= feature_no_bin_threshold:
            return
        entropy_D = entropy(D, classes)
        if entropy_D <= feature_entropy_threshold:
            return

        best_et = 1
        best_split = -1
        for i in range(1, len(D)):
            # if D[:,0][i-1]==D[:,0][i]:
            #    continue
            split = (D[:, 0][i - 1] + D[:, 0][i]) / 2

            part1 = D[np.where(D[:, 0] <= split)[0]]
            part2 = D[np.where(D[:, 0] > split)[0]]

            e1 = entropy(part1, classes)
            e2 = entropy(part2, classes)

            w1 = len(part1) / len(D)
            w2 = len(part2) / len(D)
            et = w1 * e1 + w2 * e2
            if et <= best_et:
                best_et = et
                best_split = split

        part1 = D[np.where(D[:, 0] <= best_split)[0]]
        part2 = D[np.where(D[:, 0] > best_split)[0]]
        if len(part1) == 0 or len(part2) == 0:
            return
        range_results.append(best_split)
        recursive_supervised_discretization(part1)
        recursive_supervised_discretization(part2)

    recursive_supervised_discretization(D)

    range_results.append(float(min(D[:, 0])))
    range_results.append(float(max(D[:, 0])) + 1)
    range_results = sorted(range_results)

    final_E = 0
    ranges_in_nice_format = ""
    for i in range(len(range_results) - 1):
        start = range_results[i]
        end = range_results[i + 1]
        ranges_in_nice_format += "[ " + str(start) + ', ' + str(end) + ")"
        Z = D[np.where(((D[:, 0] >= start) & (D[:, 0] < end)))[0]]
        # print(Z)
        e = entropy(Z, classes)
        # print(start,end,len(Z),e)
        final_E += (len(Z) / len(D)) * e
    print("Final entropy", final_E)
    print("discretized ranges", ranges_in_nice_format)


def get_feature_class(feat):
    return np.array(df[[feat,'target']].sort_values(feat))


if __name__ == '__main__':
    # X = np.array([[10, -1], [15, -1], [18, 1], [19, 1], [24, -1], [29, 1], [30, 1], [31, 1], [40, -1], [44, -1], [55, -1], [64, -1]])
    target_classes = [0, 1, 1, 0, 0, 1]
    df = pd.read_csv("selected_good_fit_features.csv")
    dislocs = ['sigma_110_e', 'sigma_110_s', 'sigma_112_e', 'sigma_112_s', 'sigma_123_e', 'sigma_123_s']
    features = ['Nb', 'Ta', 'Ti', 'Mo', 'Hf', 'Zr', 'eta', 'a0', 'C11', 'C12', 'C44', 'Ac',
                'Ecoh_(eV)', 'gamma_110_usf', 'gamma_112_usf', 'gamma_123_usf', 'tao_110_iss', 'tao_112_iss', 'tao_123_iss']
    df['target'] = df['target'].apply(lambda x: dislocs.index(x) if x in dislocs else x)
    df['feature_1'] = df['feature_1'].apply(lambda x: features.index(x) if x in features else x)
    df['feature_2'] = df['feature_2'].apply(lambda x: features.index(x) if x in features else x)
    df['feature_3'] = df['feature_3'].apply(lambda x: features.index(x) if x in features else x)


    
    columns = list(df.columns)
    columns.remove('target')
    for c in columns:
        print("Discretization for feature: {}".format(c))
        X = np.array(df[[c,'target']].sort_values(c))

        supervised_discretization(X, classes=target_classes, feature_entropy_threshold=0.3,
                                  feature_no_bin_threshold=10)
        print('=' * 100)

