# Author: Arjun S Kulathuvayal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [6, 6]


def FI_plot(fi_df, target):
    #fi_df = fi_df[fi_df['importances'] >= 0.01]
    fig, ax1 = plt.subplots()
    plt.bar(fi_df['feature'], fi_df['importances'], yerr=fi_df['std_dev'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title(f"Feature importance | Target: {target}")
    ax1.set_ylabel("Mean accuracy decrease")
    ax1.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax1.tick_params(axis='x', bottom=False, top=False, right=False)
    ax1.tick_params(axis='y', direction='in')
    #ax2 = ax1.twinx()
    #ax2.plot(range(0, len(feature_names)), df.drop(['D'], axis=1).std().tolist(), '--o', color="#2E4068", label="Standard deviation")
    #ax2.set_ylabel("Standard deviation")
    #ax2.tick_params(axis='x', labelrotation=0, labelsize=10)
    # ax2.tick_params(direction='in')
    fig.tight_layout()
    combo = '_'.join(fi_df['feature'])
    plt.savefig(f'RF/{target}/{combo}.png')
    plt.close()
    #print(f"Target '{target}' | Feature importance plot saved as 'RF/{target}/Feature_importances.png' for feature combinations {fi_df['feature'].to_list()}")


def Feature_importances_LoLo_RF(df, target):
    y = df[target].to_numpy()
    X = df.drop(columns=target).to_numpy()

    imp_1, imp_2, imp_3, s_1, s_2, s_3 = [], [], [], [], [], []
    for i in range(12):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LoLoRF()
        model.fit(X_train, y_train)
        result = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=42)
        imp_1.append(result.importances_mean[0])
        imp_2.append(result.importances_mean[1])
        imp_3.append(result.importances_mean[2])
        s_1.append(result.importances_std[0])
        s_2.append(result.importances_std[1])
        s_3.append(result.importances_std[2])

    print(df)
    plt.bar(df.Mo, df.a0)
    plt.show()
    exit()


    feature_names = df.drop([target], axis=1).columns.tolist()
    forest_importances = pd.DataFrame(data={'feature': feature_names, 'importances': [np.average(imp_1), np.average(imp_2), np.average(imp_3)],
                                            'std_dev': [np.average(s_1), np.average(s_2), np.average(s_3)]})
    os.makedirs(f'RF/{target}', exist_ok=True)
    combo = '_'.join(forest_importances['feature'])
    forest_importances.to_csv(f'RF/{target}/{combo}.csv', index=False)
    FI_plot(forest_importances, target)

    top_features = forest_importances.nlargest(3, "importances")
    feature_importance_dict = dict(zip(top_features["feature"], top_features["importances"]))

    return feature_importance_dict


def features_and_element_labels_df():
    trgt_df_ = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df_.iloc[:, 0:7]
    ftrs_df = pd.read_csv("Table_A1_MPEAs_Properties.csv")
    elements = ['Nb', 'Ta', 'Ti', 'Mo', 'Hf', 'Zr']
    data = []
    for alloy in trgt_df.Composition:
        row = {element: (1 if element in alloy else 0) for element in elements}
        data.append(row)

    elements_df = pd.DataFrame(data)
    # elements_df.insert(loc=0, column='Composition', value=trgt_df_['Composition'].tolist())
    # elements_df.to_csv("elements_df.csv", index=False)
    df = pd.concat([elements_df, ftrs_df], axis=1)
    return df


def main():
    df = pd.read_csv('selected_good_fit_features.csv')
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    sample_df = features_and_element_labels_df()
    sample_df.to_csv('temp_.csv', index=False)
    trgt_df = trgt_df.iloc[:, 1:7]

    analysis_df = pd.DataFrame(columns=['target', 'feature_1', 'feature_2', 'feature_3', 'avg_R2', 'FF', 'FFS', 'SF', 'SFS', 'TF', 'TFS'])
    for eachTarget in trgt_df.columns:
        each_targ_df = df[df['target'] == eachTarget]
        each_targ_df = each_targ_df[(each_targ_df['avg_R2'] >= 0.71) & (each_targ_df['avg_R2'] <= 1)]
        if each_targ_df.shape[0] > 0:
            for i, row in each_targ_df.iterrows():
                df_ = pd.read_csv('temp_.csv', usecols=[row['feature_1'], row['feature_2'], row['feature_3']])
                df__ = trgt_df[eachTarget]
                df___ = pd.concat([df_, df__], axis=1)
                best_ftr_ifo = Feature_importances_LoLo_RF(df___, eachTarget)

                analysis_df.loc[len(analysis_df)] = [eachTarget, row['feature_1'], row['feature_2'], row['feature_3'],
                                                     row['avg_R2'], list(best_ftr_ifo.keys())[0], list(best_ftr_ifo.values())[0],
                                                     list(best_ftr_ifo.keys())[1],  list(best_ftr_ifo.values())[1],
                                                     list(best_ftr_ifo.keys())[2],  list(best_ftr_ifo.values())[2]]
                print(analysis_df)

    analysis_df.to_csv('final_analysis.csv', index=False)
    print("Task done")


if __name__ == '__main__':
    main()