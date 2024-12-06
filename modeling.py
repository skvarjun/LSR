# Author: Arjun S Kulathuvayal.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['text.usetex'] = True
pd.set_option('display.max_columns', 26)
pd.set_option('display.width', 10000)

def analysis(data):
    ftrs_df  = pd.read_csv("Table_A1_MPEAs_Properties.csv")
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df.iloc[:, 1:7]
    for i in trgt_df.columns:
        df = pd.concat([trgt_df[i], data, ftrs_df], axis=1)
        print(df)
        exit()


def main():
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df.iloc[:, 0:6]

    elements = ['Nb', 'Ta', 'Ti', 'Mo', 'Hf', 'Zr']
    data = []
    for alloy in trgt_df.Composition:
        row = {element: (1 if element in alloy else 0) for element in elements}
        data.append(row)

    elements_df = pd.DataFrame(data)

    return elements_df


if __name__ == '__main__':
    data = main()
    analysis(data)
