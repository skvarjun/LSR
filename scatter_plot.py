# Author: Arjun S Kulathuvayal.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['text.usetex'] = True


def main():
    ftrs_df = pd.read_csv("Table_A1_MPEAs_Properties.csv", usecols=[3, 4, 5])
    trgt_df = pd.read_csv("Table_A2_MPEAs_Dislocations.csv")
    trgt_df = trgt_df.iloc[:, 0:6]

    df = pd.concat([ftrs_df, trgt_df['eta_110_e']], axis=1)

    fig = px.scatter_matrix(df, dimensions=df.columns.tolist(), title='Scatter Matrix for All Columns')
    fig.write_image('scatter_plot.png', width=1000, height=1000)




if __name__ == '__main__':
    main()
