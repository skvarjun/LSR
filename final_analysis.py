# Author: Arjun S Kulathuvayal.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
from pycirclize import Circos


def graph(disloc, mech_pro):
    sectors = {}
    for i, dis in enumerate(disloc):
        sectors[dis] = len(mech_pro[i])

    circos = Circos(sectors, space=2)

    frequencies = {}
    for j, disl in enumerate(mech_pro):
        frequencies[disloc[j]] = list(disl.values())

    # Add tracks and plot histograms
    for i, sector in enumerate(circos.sectors):
        sector_name = sector.name
        freq_data = frequencies[sector_name]
        track = sector.add_track((50, 100))
        track.axis(fc="lightgrey")
        track.bar(np.arange(len(freq_data)) + 0.5, freq_data, color="deepskyblue")

        sector.text(list(frequencies.keys())[i], adjust_rotation=True, fontsize=12)

    # Save and display the plot
    circos.savefig("circular_histogram.pdf")
    plt.show()


def fs_range(df):
    FF_threshold = df['FFS'].quantile(0.5)
    SF_threshold = df['SFS'].quantile(0.5)
    TF_threshold = df['TFS'].quantile(0.5)
    if FF_threshold < 0:
        FF_threshold = 0
    elif SF_threshold < 0:
        SF_threshold = 0
    elif TF_threshold < 0:
        TF_threshold = 0
    return FF_threshold, SF_threshold, TF_threshold

def Main():
    df = pd.read_csv('final_analysis.csv')
    dislocations = ['sigma_110_s', 'sigma_112_e', 'sigma_112_s', 'sigma_123_e']
    mech_properties = []
    for eachDis in dislocations: #
        data = df[df['target'] == eachDis]
        df_ = pd.DataFrame(columns=['FF', 'FFS', 'SF', 'SFS', 'TF', 'TFS'])
        for i, row in data.iterrows():
            first_ftr_wgtd_score = row.avg_R2 * row.FFS
            second_ftr_wgtd_score = row.avg_R2 * row.SFS
            third_ftr_wgtd_score = row.avg_R2 * row.TFS
            df_.loc[len(df_)] = [row.FF, first_ftr_wgtd_score, row.SF, second_ftr_wgtd_score, row.TF, third_ftr_wgtd_score]
        FFS_thresh, SFS_thresh, TFS_thresh = fs_range(df_)

        df_FFS = df_[(df_['FFS'] >= FFS_thresh)]
        df_SFS = df_[(df_['SFS'] >= SFS_thresh)]
        df_TFS = df_[(df_['TFS'] >= TFS_thresh)]
        important_FF = df_FFS.groupby('FF', as_index=False)['FFS'].mean()
        important_SF = df_SFS.groupby('SF', as_index=False)['SFS'].mean()
        important_TF = df_TFS.groupby('TF', as_index=False)['TFS'].mean()
        # print(important_FF)
        # print(important_SF)
        # print(important_TF)

        if eachDis == 'sigma_112_e':
            dict1, dict2 = dict(zip(important_FF['FF'], important_FF['FFS'])), dict(zip(important_SF['SF'], important_SF['SFS']))
            merged_dict = {}
            for key, value in {**dict1, **dict2}.items():
                if key in dict1 and key in dict2:
                    merged_dict[key] = max(dict1[key], dict2[key])
                else:
                    merged_dict[key] = value

            print(eachDis, merged_dict)
            mech_properties.append(merged_dict)
        else:
            dict1, dict2, dict3 = dict(zip(important_FF['FF'], important_FF['FFS'])), dict(
                zip(important_SF['SF'], important_SF['SFS'])), dict(zip(important_TF['TF'], important_TF['TFS']))
            merged_dict = {}
            for key in set(dict1) | set(dict2) | set(dict3):  # Union of all keys
                merged_dict[key] = max(
                    dict1.get(key, float('-inf')),
                    dict2.get(key, float('-inf')),
                    dict3.get(key, float('-inf'))
                )

            print(eachDis, merged_dict)
            mech_properties.append(merged_dict)
    print("-----------------------------------------------------------------")
    graph(dislocations, mech_properties)


if __name__ == "__main__":
    Main()
