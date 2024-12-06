# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['text.usetex'] = True
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 10000)
import seaborn as sns


def heatmap(data, target):
    df = pd.DataFrame(data)
    correlation_matrix = df.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 10},
        cbar=True
    )
    plt.tight_layout()
    plt.savefig(f"heatmap/{target}.pdf")
    plt.show()

def main():
    ftrs_df = pd.read_csv("Table_A1_MPEAs_Properties.csv")
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df.iloc[:, 1:7]

    for trgt in trgt_df.columns:
        df = pd.concat([trgt_df, ftrs_df.drop(columns=['Composition'])], axis=1)
        print(df)
        heatmap(df, trgt)
        exit()

if __name__ == '__main__':
    main()
