import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
import time
import numpy as np
import seaborn as sns
from scipy.signal import argrelextrema


def Create_EKG_DF(ekgs):
    ekg_df = pd.DataFrame()
    prog_bar = st.progress(0)
    for i in range(len(ekgs)):
        file = dir+'/'+ekgs[i]
        prog_bar.progress(i/len(ekgs))
        example = pd.read_csv(file)
        dob = example.loc['Date of Birth', 'Name']
        date = example.loc['Recorded Date', 'Name']
        classification = example.loc['Classification', 'Name']
        version = example.loc['Software Version', 'Name']
        ekg_df.loc[i, 'name'] = ekgs[i]
        ekg_df.loc[i, 'date'] = date
        ekg_df.loc[i, 'clas'] = classification
        ekg_df.loc[i, 'vers'] = version
    ekg_df['day'] = ekg_df.date.str[0:10]
    ekg_df.date = pd.to_datetime(ekg_df.date)
    ekg_df.sort_values(by='date', inplace=True)

    # st.write('I have finished writing EKGs.csv. Try another function!')
    # st.write(ekg_df)
    return ekg_df


def Get_EKG(name):
    file = dir+'/'+name
    df = pd.read_csv(file)
    # st.write(df)
    return df


def Clean_EKG(ekg):
    ekg = ekg[1000::]
    ekg.reset_index(inplace=True, drop=False)
    ekg.columns = ['micro_volts', 'ignore']
    ekg.micro_volts = ekg.micro_volts.astype(float)
    ekg['seconds'] = ekg.index * 1/510.227
    ekg = ekg[['micro_volts', 'seconds']]
    ekg['peak'] = ekg.micro_volts - ekg.micro_volts.shift(-7)
    ekg['interval'] = ekg.micro_volts - ekg.micro_volts.shift(-1)
    # ekg = Get_R_Peaks(ekg)
    ekg = Get_alt_r(ekg)
    return ekg


def Get_R_Peaks(ekg):
    # identify QRS complexes
    ekg['qrs'] = 0
    # size = st.sidebar.slider('first pass size', min_value=1, max_value=35, value=5)
    for i in range(ekg.shape[0]):
        numbers = ekg.interval[i-4:i+4]
        if numbers.max()-numbers.min() > 50:
            ekg.loc[i, 'qrs'] = 1
    # # create 3 empty cols
    ekg['int_peak'] = 0
    ekg['int_peak_2'] = 0
    ekg['r_peak'] = 0
    # identify possible r peaks
    qrs_indices = ekg[ekg.qrs == 1].index.tolist()
    # get interim peak
    for idx in qrs_indices:
        diffs = ekg.micro_volts[idx-5:idx+5]
        if diffs.max()-diffs.min() > 500:
            ekg.loc[diffs.idxmax(), 'int_peak'] = 1

    int_peak_df = ekg[ekg.int_peak == 1]
    int_peak_indices = int_peak_df.index

    for idx in int_peak_indices.tolist():
        calc_bottom = idx-50
        abs_bottom = np.min(int_peak_indices)
        calc_top = idx+50
        abs_top = np.max(int_peak_indices)
        if calc_bottom > abs_bottom:
            start = calc_bottom
        else:
            start = abs_bottom
        if calc_top > abs_top:
            end = abs_top
        else:
            end = calc_top

        diffs = ekg.micro_volts[start:end]
        ekg.loc[diffs.idxmax(), 'r_peak'] = 1

    ekg = ekg[['micro_volts', 'seconds', 'peak', 'interval', 'qrs', 'r_peak']]
    return ekg


def Get_alt_r(ekg):
    n = 190
    ekg['int_peak'] = ekg.iloc[argrelextrema(ekg.micro_volts.values, np.greater_equal,
                                             order=n)[0]]['micro_volts']
    med = ekg.int_peak.median()
    ekg.int_peak = np.where(ekg.micro_volts < med*.55, 0, ekg.int_peak)
    ekg['r_peak'] = np.where(ekg.int_peak > 0, 1, 0)
    ekg = ekg[['micro_volts', 'seconds', 'interval', 'r_peak']]
    return ekg


def Get_Singles(ekg):
    singles = ekg[ekg.r_peak == 1]
    return singles


def Get_PACs(singles):
    singles['interval'] = singles.seconds.shift(-1) - singles.seconds
    median = singles.interval.median()
    singles['med'] = median
    singles['sq_diff'] = (singles.med-singles.interval)*(singles.med-singles.interval)
    PACs = int((singles[singles.sq_diff > .015].shape[0]/2)+.5)
    # temporary visualization for dev

    # if PACs > 0:
    #     st.write(idx, PACs, singles.sq_diff)
    #     fig, ax = plt.subplots()
    #     plt.plot(ekg.index, ekg.micro_volts)
    #     for i in singles.index.tolist():
    #         plt.vlines(i, ymin=1000, ymax=1300, colors='r')
    #     st.pyplot(fig)

    ekg_df.to_csv('EKGs.csv', index=False)
    return PACs


def Get_Rate(singles):
    singles['r_interval'] = singles.seconds - singles.seconds.shift(1)
    med = singles.r_interval.median()
    rate = int(58.3/med)
    return rate


def Cull_Dense_R_Peak(ekg):
    """If there is too much non-peak activity within range of r_peaks this
    will tell the caller to cull the EKG from the db
    """
    r_peak_med = ekg[ekg.r_peak == 1]['micro_volts'].median()
    r_peak_density = round(ekg[ekg.micro_volts > r_peak_med*.9]['r_peak'].mean(), 2)
    if r_peak_density >= .24:
        # st.write('this is okay', r_peak_density)
        return False
    else:
        # st.write('this would be culled', r_peak_density)
        return True


# create streamlit page
path = './'
dir = path + 'electrocardiograms'
ekgs = os.listdir(dir)
st.set_page_config(layout="wide")
if os.path.isfile('EKGs.csv'):
    index = 0
else:
    index = 1
function = st.sidebar.selectbox(
    'Select a Function', ['Show an EKG', 'Reset EKG Database',  'Show PACs Over Time'], index=index)
ekg_df = pd.read_csv('EKGs.csv')

#############skip for now#################
if function == 'Reset EKG Database':
    a = st.empty()
    a.write(f'I am creating an index of your {len(ekgs)} EKGs...')
    ekg_df = Create_EKG_DF(ekgs)
    # poor = ekg_df[ekg_df.clas=='Poor Recording']
    ekg_df = ekg_df[~ekg_df.clas.str.contains('Poor Recording')]
    ekg_df.to_csv('EKGs.csv', index=False)
    st.write(ekg_df)
    a.write(
        f'I have finished writing {ekg_df.shape[0]} EKGs with good recordings to EKGs.csv. Try another function!')
##########################################
elif function == 'Show PACs Over Time':
    ekg_df = pd.read_csv('EKGs.csv')
    ekg_df.reset_index(inplace=True, drop=True)

    if 'PACs' not in ekg_df.columns:
        a = st.empty()
        b = st.empty()
        a.write(f'I am working your list of {ekg_df.shape[0]} EKGs with good recordings.')
        prog_bar = st.progress(0)
        for idx, row in ekg_df.iterrows():
            prog_bar.progress((idx)/ekg_df.shape[0])
            ekg_str = ekg_df.loc[idx, 'name']
            ekg = Get_EKG(ekg_str)
            # st.write(ekg_df)
            clas = ekg_df.loc[idx, 'clas']
            if clas in ['Atrial Fibrillation', 'Heart Rate Over 150', 'Inconclusive']:
                ekg_df.loc[idx, 'PACs'] = None
            else:
                this_classification = ekg_df.loc[ekg_df[ekg_df.name == ekg_str].index.tolist()[
                    0], 'clas']
                b.write(f'I am working {ekg_str}')
                ekg = Clean_EKG(ekg)
                singles = Get_Singles(ekg)
                PACs = Get_PACs(singles)
                if Cull_Dense_R_Peak(ekg):
                    ekg_df.loc[idx, 'PACs'] = None
                else:
                    ekg_df.loc[idx, 'PACs'] = PACs
        prog_bar.empty()
        b.empty()
        a.empty()
    else:
        pass
    pos_PACs = ekg_df[ekg_df.PACs > 0].shape[0]
    not_null = ekg_df[ekg_df.PACs.notnull()].shape[0]
    # set days for dataset
    first_day = ekg_df.date.min()
    last_day = ekg_df.date.max()
    # create list of days for x_axis
    x_range = pd.DataFrame(pd.date_range(first_day, last_day, freq='d'))
    x_range.columns = ['date']
    x_range.date = x_range.date.astype(str)
    x_range['day'] = x_range.date.str[0:10]
    x_range.reset_index(inplace=True, drop=True)
    x_range = x_range['day']

    afib = ekg_df[ekg_df.clas == 'Atrial Fibrillation']
    afib['day'] = afib.date.str[0:10]
    afib.day = pd.to_datetime(afib.day)
    afib['afib'] = 1
    afib.drop_duplicates(subset='day', inplace=True)
    afib = afib[['afib', 'day']]

    temp = ekg_df.groupby(by='day').max()
    temp['day'] = temp.date.str[0:10]
    temp.reset_index(inplace=True, drop=True)

    plot_df = pd.merge(temp, x_range, on='day', how='outer')
    plot_df.day = pd.to_datetime(plot_df.day)
    plot_df.sort_values(by='day', inplace=True)
    plot_df.PACs.fillna(0, inplace=True)
    plot_df.PACs = plot_df.PACs.astype(int)

    export = pd.merge(plot_df, afib, on='day', how='outer')
    export.afib = export.afib.fillna(0)
    export.afib = export.afib.astype(int)
    export = export[['day', 'PACs', 'afib']]

    how = st.sidebar.radio('How to Plot PACs', ['Bar', 'Rolling Mean'])

    fig, ax = plt.subplots(figsize=(18, 8))

    if how == 'Bar':
        plt.bar(export.day, export.PACs)
    else:
        n = st.sidebar.slider('Number of Days Rolling', min_value=1, max_value=30, value=5)
        plot_df['avg'] = plot_df.PACs.rolling(window=n).mean()
        plt.plot(plot_df.day, plot_df.avg)
    ax.set_xticks(export.day[-1::-20], label=export.day[-1::-20])
    plt.xticks(rotation=70, ha='right')
    title_fontdict = {'fontsize': 24, 'fontweight': 10}
    label_fontdict = {'fontsize': 20, 'fontweight': 8}
    ax.set_ylabel('Number of PACs', fontdict=label_fontdict)
    if afib.shape[0] > 0:
        for day in list(set(afib.day.tolist())):
            plt.vlines(day, 0, 15, colors='r', alpha=.5)
        ax.set_title(
            f'Maximum PACs in 30 Seconds in {pos_PACs} out of {not_null} eligible EKGs by Date - Days with AFib in Red', fontdict=title_fontdict)
    else:
        ax.set_title(
            f'Maximum PACs in 30 Seconds in {pos_PACs} out of {not_null} eligible EKGs by Date', fontdict=title_fontdict)
    st.pyplot(fig)
    export.rename(columns={'day': 'date'}, inplace=True)
    st.write('Export file for this figure is EKG_by_day.csv')
    export.to_csv('EKG_by_day.csv', index=False)
##########################################
elif function == 'Show an EKG':
    # selection
    year = st.sidebar.selectbox('Year of EKG', ['2019', '2020', '2021', '2022'], index=1)
    month = st.sidebar.selectbox(
        'Month of EKG', ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
    ekgs = ekg_df[ekg_df.name.str.contains(year+'-'+month)]
    type = st.sidebar.selectbox('Classification', list(set(ekgs.clas.tolist())))
    show_df = ekgs[ekgs.clas == type]
    ekg_str = st.sidebar.selectbox('Choose an EKG', show_df.name)
# select and clean EKG to show
    ekg = Get_EKG(ekg_str)
    # get the classification and PAC number for this ekg from ekg_df
    this_classification = ekg_df.loc[ekg_df[ekg_df.name == ekg_str].index.tolist()[0], 'clas']
    this_PACs = ekg_df.loc[ekg_df[ekg_df.name == ekg_str].index.tolist()[0], 'PACs']

    st.write(f'You have selected {ekg_str}, classified as {this_classification}')
    ekg = Clean_EKG(ekg)

    # get singles and rate
    singles = Get_Singles(ekg)
    rate = Get_Rate(singles)

    # plot EKG
    x = ekg.seconds
    y = ekg.micro_volts
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_ylim(y.min(), y.max())

    # set PACs and level of
    if pd.isna(this_PACs):
        PACs = None
        level = 0
    else:
        PACs = int(this_PACs)
        level = int(round(3*PACs/14, 0))
    color_palette = sns.color_palette('RdYlGn_r')

    if type == 'Atrial Fibrillation':
        ax.set_facecolor(color_palette[5])
    elif type == 'Inconclusive':
        ax.set_facecolor(color_palette[4])
    elif type == 'Heart Rate Over 120':
        ax.set_facecolor(color_palette[4])
    elif type == 'Heart Rate Under 50':
        ax.set_facecolor(color_palette[4])
    elif type == 'Heart Rate Over 150':
        ax.set_facecolor(color_palette[5])
    else:
        ax.set_facecolor(color_palette[level])
    # set title and labels
    if pd.isna(this_PACs):
        ax.set_title(f'The EKG appears to have a rate of {rate}. It cannot be used to judge PACs.')
        # st.write(f'The EKG appears to have a rate of {rate}. It cannot be used to judge PACs.')
    else:
        # st.write(f'The EKG evidences {PACs} PACs with a heart rate of {rate}')
        ax.set_title(f'The EKG evidences {PACs} PACs with a heart rate of {rate}')
    ax.set_xlabel('Seconds')
    ax.yaxis.set_visible(False)
    plt.plot(x, y)
    st.pyplot(fig)
