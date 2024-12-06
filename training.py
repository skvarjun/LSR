# Author: Arjun S Kulathuvayal.
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from lolopy.learners import RandomForestRegressor as LoLoRF
font = {'family': 'serif',
        'weight': 'normal',
        'size': 10}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 4.5]
pd.set_option('display.max_columns', 26)
pd.set_option('display.width', 10000)
import itertools


def Results(data, target):
    y = data[target].to_numpy()
    X = data.drop(target, axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LoLoRF()
    model.fit(X_train, y_train)

    #cv_prediction = cross_val_predict(model, X_test, y_test, cv=KFold(5, shuffle=True))

    LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)
    LoLo_r2 = r2_score(y_test, LoLo_y_pred)
    y_resid = LoLo_y_pred - y_test


    LoLo_mse = mean_squared_error(y_test, LoLo_y_pred)
    LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)
    #print(f"LOLO R2: {LoLo_r2}, MSE: {LoLo_mse}, MAE: {LoLo_mae}, Residuals: {np.sqrt(np.power(LoLo_y_pred - y_test, 2).mean())}")
    # for scorer in ['r2_score', 'mean_absolute_error', 'mean_squared_error']:
    #     score = getattr(metrics, scorer)(y_test, LoLo_y_pred)
    #     print(scorer, score)

    fig, axs = plt.subplots(1, 2)
    # axs[0].hist2d(pd.to_numeric(y_test), cv_prediction, norm=LogNorm(), bins=64, cmap='Blues', alpha=0.9)
    # axs[0].scatter(pd.to_numeric(y_test), LoLo_y_pred, s=5)
    axs[0].errorbar(y_test, LoLo_y_pred, LoLo_y_uncer, fmt='o', ms=2.5, ecolor='gray')
    axs[0].set_xlabel(f'Actual {target}')
    axs[0].set_ylabel(f'Predicted {target}')
    axs[0].plot(y_test, y_test, '--', color='gray')
    axs[0].set_title('RF Performance : R2 = {:.2g}, MSE = {:.3g}'.format(LoLo_r2, LoLo_mse),
                     fontsize=10, pad=15)
    axs[0].legend([f'Test size : {len(y_test)}', f'Train size : {len(y_train)}'], markerscale=0, handlelength=0,
                  loc='upper left')

    x = np.linspace(-6, 6, 50)
    conv_resid = np.divide(y_resid, np.sqrt(np.power(y_resid, 2).mean()))
    axs[1].hist(conv_resid, x, density=True)  # conventional cross validation error => constant error (MSE) estimation for all points
    axs[1].plot(x, norm.pdf(x), 'k--', lw=1.00)  # probability density function (pdf) of the normal distribution for data points x
    axs[1].set_title('Uncertainty', fontsize=10, pad=15)
    axs[1].set_ylabel('Probability Density')
    axs[1].set_xlabel('Normalized residual')
    fig.suptitle(f'Target: {target}')
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    plt.savefig(f'RF/{target}results.png')
    plt.close()
    return LoLo_r2, LoLo_mse


def analysis(data):
    ftrs_df  = pd.read_csv("Table_A1_MPEAs_Properties.csv")
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df.iloc[:, 1:7]

    selected_ftr_df = pd.DataFrame(columns=['target'] + [f"feature_{i+1}" for i in range(3)] + ['avg_R2', 'avg_MSE'])
    for i in trgt_df.columns:
        df = pd.concat([trgt_df[i], data, ftrs_df], axis=1).drop(columns=['Composition'])
        df.to_csv("temp_.csv")
        all_features = df.drop(columns=i).columns.tolist()
        combinations = list(itertools.combinations(all_features, 3))
        for combination in combinations:
            df = pd.read_csv("temp_.csv", usecols=[i] + list(combination))
            R2s,MSEs = [], []
            for j in range(12):
                R2, MSE = Results(df, target=i)
                R2s.append(R2)
                MSEs.append(MSE)
            print(np.average(R2s), np.average(MSEs), combination)
            if np.average(R2s) > 0 and np.average(R2s) < 1:
                selected_ftr_df.loc[len(selected_ftr_df)] = [i] + list(combination) + [np.average(R2s), np.average(MSEs)]
    selected_ftr_df.to_csv("selected_good_fit_features.csv", index=False)


def graph():
    R2_set = None
    plt.rcParams["figure.figsize"] = [10, 6]
    num_groups = len(R2_set)
    num_bars_per_group = len(R2_set[0])

    group_indices = np.arange(num_groups)
    bar_width = 0.150
    bar_offsets = np.arange(num_bars_per_group) * bar_width

    fig, ax = plt.subplots()
    for i, (key, values) in enumerate(R2_set.items()):
        x_positions = group_indices[i] + bar_offsets
        bars = ax.bar(x_positions, values, width=bar_width, label=f'Group {key}')
        sigmas = [r'$\sigma^{110}_e$', r'$\sigma^{110}_s$', r'$\sigma^{110}_e$', r'$\sigma^{112}_s$',
                  r'$\sigma^{123}_e$', r'$\sigma^{123}_s$']
        k = 0
        for bar, value in zip(bars, values):
            y_position = value if value >= 0 else value - 0.05  # Adjust position for negative values
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_position,
                f'{sigmas[k]}',  # Format to 2 decimal places
                ha='center',
                va='bottom' if value >= 0 else 'top',
                fontsize=8
            )
            k += 1
    ax.set_xticks(group_indices + bar_width * (num_bars_per_group - 1) / 2)
    ax.set_xticklabels([f'Trial {key + 1}' for key in R2_set.keys()])
    ax.set_xlabel('Trials')
    ax.set_ylabel(r'$R^2$ score')
    # ax.legend()
    plt.title("Training with Random Forest")
    plt.tight_layout()
    plt.show()


def main():
    trgt_df_ = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df_.iloc[:, 0:7]

    elements = ['Nb', 'Ta', 'Ti', 'Mo', 'Hf', 'Zr']
    data = []
    for alloy in trgt_df.Composition:
        row = {element: (1 if element in alloy else 0) for element in elements}
        data.append(row)

    elements_df = pd.DataFrame(data)
    # elements_df.insert(loc=0, column='Composition', value=trgt_df_['Composition'].tolist())
    # elements_df.to_csv("elements_df.csv", index=False)
    return elements_df

if __name__ == '__main__':
    data = main()
    analysis(data)
