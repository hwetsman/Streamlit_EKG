import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
import time
import numpy as np
import seaborn as sns
from scipy.signal import argrelextrema
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def Create_EKG_DF(ekgs):
    years = ['2019', '2020', '2021', '2022']
    ekg_df = pd.DataFrame()
    prog_bar = st.progress(0)
    ekg_df = pd.DataFrame()
    for year in years:
        # st.write(year)
        ekgs = [x for x in os.listdir(f'{path}electrocardiograms_{year}') if x != '.DS_Store']

        # st.write(len(ekgs))
        for i in range(len(ekgs)):
            #     # st.write(i)
            temp = pd.DataFrame()
            file = f'{path}electrocardiograms_{year}/{ekgs[i]}'
            prog_bar.progress(i/len(ekgs))
            # st.write(file)
            example = pd.read_csv(file)
            # st.write(example)
            dob = example.loc['Date of Birth', 'Name']
            date = example.loc['Recorded Date', 'Name']
            classification = example.loc['Classification', 'Name']
            version = example.loc['Software Version', 'Name']
            ekg_df.loc[f'{year}_{i}', 'name'] = ekgs[i]
            ekg_df.loc[f'{year}_{i}', 'date'] = date
            ekg_df.loc[f'{year}_{i}', 'clas'] = classification
            ekg_df.loc[f'{year}_{i}', 'vers'] = version
            ekg_df.loc[f'{year}_{i}', 'dir'] = f'{path}electrocardiograms_{year}'
            # st.write(ekg_df)
    ekg_df['day'] = ekg_df.date.str[0:10]
    ekg_df.date = pd.to_datetime(ekg_df.date)
    ekg_df.sort_values(by='date', inplace=True)
    # st.write(ekg_df)
    ekg_df = ekg_df.reset_index(drop=True)
    # st.write('I have finished writing EKGs.csv. Try another function!')
    # st.write(ekg_df)
    return ekg_df


def Get_EKG(name, year):
    file = f'{path}electrocardiograms_{year}/{name}'
    # file = dir+'/'+name
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

# QRS complexes happen in less than 0.05 seconds and are at least 500 microvolts in difference
# 0.002 seconds between readings so we need 25-row rolling frames


def Find_QRS(ekg):
    """
    This is an attempt at an improved QRS finder. The problem is finding the correct
    parameters of rolling window and microvolt difference in that window. It remains
    unused and experimental.
    """
    ekg1 = ekg.copy()
    ekg1['qrs'] = 0
    ekg1['rolling_min'] = ekg1.micro_volts.rolling(25, center=True).min()
    ekg1['rolling_max'] = ekg1.micro_volts.rolling(25, center=True).max()
    ekg1['diff'] = ekg1.rolling_max-ekg1.rolling_min
    ekg1['waste'] = 0
    ekg1['qrs'] = np.where(ekg1['diff'] > 500, 1, 0)
    # st.write(np.where(ekg1['diff'] > 500, 1, 0))
    # st.write(ekg1)


def Get_R_Peaks(ekg):
    # identify QRS complexes
    ekg['qrs'] = 0
    # size = st.sidebar.slider('first pass size', min_value=1, max_value=35, value=5)
    for i in range(ekg.shape[0]):
        # numbers = ekg.interval[i-4:i+4]
        numbers = ekg.interval[1-12:i+12]
        # if numbers.max()-numbers.min() > 50:
        if numbers.max()-numbers.min() > 500:
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
    # n = 125
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
    # PACs = int(singles[singles.interval < .6*median].shape[0])
    # temporary visualization for dev

    # if PACs > 0:
    #     st.write(idx, median, singles.shape[0], PACs, singles.sq_diff)
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


def Set_Color_For_PACs(pacs):
    if pd.isna(this_PACs):
        PACs = None
        level = 0
    else:
        PACs = int(this_PACs)
        level = int(round(3*PACs/14, 0))
    return PACs, level


def Set_Background_Color(level):
    if type == 'Atrial Fibrillation':
        face_color = 5
    elif type == 'Inconclusive':
        face_color = 4
    elif type == 'Heart Rate Over 120':
        face_color = 4
    elif type == 'Heart Rate Under 50':
        face_color = 4
    elif type == 'Heart Rate Over 150':
        face_color = 5
    else:
        face_color = level
    return face_color


def Set_Title(this_PACs, rate, PACs):
    if pd.isna(this_PACs):
        title = f'The EKG appears to have a rate of {rate}. It cannot be used to judge PACs.'
    else:
        title = f'The EKG evidences {PACs} PACs with a heart rate of {rate}'
    return title


# create streamlit page
st.set_page_config(layout="wide")

# set paths
years = ['2019', '2020', '2021', '2022']
path = './'

dir = path + 'electrocardiograms'

# create list of ekgs
ekgs = []
for year in years:
    year_list = os.listdir(f'{path}electrocardiograms_{year}')
    ekgs = ekgs+year_list

if os.path.isfile('EKGs.csv'):
    index = 0
    ekg_df = pd.read_csv('EKGs.csv')
    if 'PACs' not in ekg_df.columns:
        index = 2
else:
    index = 1
function = st.sidebar.selectbox(
    'Select a Function', ['Show an EKG', 'Reset EKG Database',  'Show PACs Over Time'], index=index)

#
# #############skip for now#################
if function == 'Reset EKG Database':
    a = st.empty()
    a.write(f'I am creating an index of your {len(ekgs)} EKGs...')
    ekg_df = Create_EKG_DF(ekgs)
    # poor = ekg_df[ekg_df.clas=='Poor Recording']
    ekg_df = ekg_df[~ekg_df.clas.str.contains('Poor Recording')]
    ekg_df.to_csv('EKGs.csv', index=False)
    # st.write(ekg_df)
    a.write(
        f'I have finished writing {ekg_df.shape[0]} EKGs with good recordings to EKGs.csv. Next, select Show PACs Over Time to process the EKGs')
# ##########################################
elif function == 'Show PACs Over Time':
    ekg_df = pd.read_csv('EKGs.csv')
    ekg_df.reset_index(inplace=True, drop=True)
    if 'PACs' not in ekg_df.columns:
        a = st.empty()
        b = st.empty()
        a.write(f'I am working your list of {ekg_df.shape[0]} EKGs with good recordings.')
        prog_bar = st.progress(0)
        for idx, row in ekg_df.iterrows():
            directory = ekg_df.loc[idx, 'dir']
            # st.write(directory)
            year = directory[-4:]
            prog_bar.progress((idx)/ekg_df.shape[0])
            ekg_str = ekg_df.loc[idx, 'name']
            ekg = Get_EKG(ekg_str, year)
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
    # create calendar graph
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
    export.rename(columns={'day': 'date'}, inplace=True)

    # plotly stuff
    how = st.sidebar.radio('How to Plot PACs', ['Bar', 'Rolling Mean'])
    if how == 'Bar':
        fig = px.bar(export, x='date', y='PACs')
    else:
        n = st.sidebar.slider('Number of Days Rolling', min_value=1, max_value=30, value=5)
        plot_df['avg'] = plot_df.PACs.rolling(window=n).mean()
        fig = px.line(plot_df, x="day", y="avg")
    fig.update_traces(marker_color='blue')
    n = 0
    for day in list(set(afib.day.tolist())):
        n = n+1
        anno = str(pd.to_datetime(day).date())
        even = (n % 2 == 0)
        if even:
            posit = 'top'
        else:
            posit = 'top left'
        fig.add_vline(x=day.timestamp()*1000, line_width=2,
                      line_dash="dash", line_color="red", annotation_text=f'{anno}', annotation_position=posit)
        # fig.add_vline(x=day, line_width=2,
        #               line_dash="dash", line_color="red")  # , annotation_text=f'{anno}')
    # set title
    if afib.shape[0] > 0:
        title = f'Daily Maximum PACs in 30 Seconds in {pos_PACs} out of {not_null} eligible EKGs by Date - Days with AFib in Red'
    else:
        title = f'Daily Maximum PACs in 30 Seconds in {pos_PACs} out of {not_null} eligible EKGs by Date'
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            visible=True
    ),
        showlegend=False,
        plot_bgcolor='black'
    )
    st.plotly_chart(fig, use_container_width=True)

    # create day of week graph
    dow_afib = afib.copy().rename(columns={'day': 'date'})
    dow_afib.date = pd.to_datetime(dow_afib.date)
    dow_afib['dow'] = dow_afib.date.dt.day_name()

    afib_group = dow_afib.groupby(by='dow').sum().reset_index(drop=False)
    afib_group.dow = pd.Categorical(afib_group.dow,
                                    categories=['Monday', 'Tuesday', 'Wednesday',
                                                'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                    ordered=True)
    afib_group.sort_values('dow', inplace=True)

    dow_df = ekg_df.copy()
    dow_df['date'] = pd.to_datetime(dow_df['date'])
    dow_df['dow'] = dow_df['date'].dt.day_name()
    dow_df = dow_df.drop(['vers', 'clas', 'date', 'day'], axis=1)
    group = dow_df.groupby(by='dow').sum().rename(
        columns={'PACs': 'sum'}).reset_index(drop=False)
    count = dow_df.groupby(by='dow').count().drop('PACs', axis=1).reset_index(drop=False)

    dow_graph = pd.merge(group, count, on='dow', how='outer').rename(columns={'sum': 'total'})
    dow_graph['average'] = dow_graph.total/dow_graph.name
    dow_graph.dow = dow_graph.dow.astype('category')
    dow_graph.dow = pd.Categorical(dow_graph.dow,
                                   categories=['Monday', 'Tuesday', 'Wednesday',
                                               'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                   ordered=True)
    dow_graph.sort_values('dow', inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=dow_graph.dow, y=dow_graph.average, name="Average PACs"), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=afib_group.dow, y=afib_group.afib, name="Number of Times Afib", mode="lines"), secondary_y=True)
    title = 'Average PACs and Afib Occurances by Day of Week'
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            visible=True
    ),
        showlegend=True,
        plot_bgcolor='black'
    )
    st.plotly_chart(fig, use_container_width=True)

    # create hours graph
    hours_df = ekg_df.copy()
    hours_df['date'] = pd.to_datetime(hours_df['date'])
    hours_df['hour'] = hours_df['date'].dt.hour-1
    hours_df = hours_df.drop(['vers'], axis=1)
    hours_df.hour = hours_df['hour'].replace(-1, 23)
    afib_hour = hours_df[hours_df.clas == 'Atrial Fibrillation']

    hours_df = hours_df.drop(['name', 'date', 'clas'], axis=1)
    hours_pac_group = hours_df.groupby(by=['day', 'hour']).max().reset_index(drop=False).dropna()
    hours_pac_sum = hours_pac_group.groupby(by='hour').sum(
    ).reset_index(drop=False).rename(columns={'PACs': 'total'})
    hours_pac_max = hours_pac_group.groupby(by='hour').max(
    ).reset_index(drop=False).rename(columns={'PACs': 'maximum'})
    hours_pac_count = hours_pac_group.groupby(by='hour').count(
    ).reset_index(drop=False).rename(columns={'PACs': 'number'})
    hours_pac_graph = pd.merge(hours_pac_sum, hours_pac_count, on='hour', how='outer')
    hours_pac_graph['average'] = hours_pac_graph.total/hours_pac_graph.number

    afib_hour_group = afib_hour.groupby(by=['day', 'hour']).count().reset_index(drop=False)
    afib_hour_group['occured'] = 1
    afib_hour_graph = afib_hour_group.groupby(by='hour').count().reset_index(drop=False)
    afib_hour_graph.drop(['day', 'name', 'date', 'clas', 'PACs'], inplace=True, axis=1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(
    #     go.Bar(x=hours_pac_graph.hour, y=hours_pac_graph.average, name="Average PACs"), secondary_y=False)
    fig.add_trace(
        go.Bar(x=hours_pac_max.hour, y=hours_pac_max.maximum, name="Max PACs"), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=afib_hour_graph.hour, y=afib_hour_graph.occured, name="Number of Times Afib", mode="lines"), secondary_y=True)
    title = 'Maximum PACs/30 Seconds and Afib Occurances by Hour of Day'
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            visible=True
    ),
        showlegend=True,
        plot_bgcolor='black'
    )
    st.plotly_chart(fig, use_container_width=True)
    # group = dow_df.groupby(by='dow').sum().rename(
    #     columns={'PACs': 'sum'}).reset_index(drop=False)
    # count = dow_df.groupby(by='dow').count().drop('PACs', axis=1).reset_index(drop=False)
    #
    # dow_graph = pd.merge(group, count, on='dow', how='outer').rename(columns={'sum': 'total'})
    # dow_graph['average'] = dow_graph.total/dow_graph.name
    # dow_graph.dow = dow_graph.dow.astype('category')
    # dow_graph.dow = pd.Categorical(dow_graph.dow,
    #                                categories=['Monday', 'Tuesday', 'Wednesday',
    #                                            'Thursday', 'Friday', 'Saturday', 'Sunday'],
    #                                ordered=True)
    # dow_graph.sort_values('dow', inplace=True)
    #
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(
    #     go.Bar(x=dow_graph.dow, y=dow_graph.average, name="Average PACs"), secondary_y=False)
    # fig.add_trace(
    #     go.Scatter(x=afib_group.dow, y=afib_group.afib, name="Number of Times Afib", mode="lines"), secondary_y=True)
    # title = 'Average PACs and Afib Occurances by Day of Week'
    # fig.update_layout(title_text=title, title_x=0.5, margin=dict(l=0, r=0, t=30, b=0),
    #                   xaxis=dict(
    #     showline=False,
    #     showgrid=False,
    #     showticklabels=True,
    #     linecolor='rgb(204, 204, 204)',
    #     linewidth=2,
    #     ticks='outside',
    #     tickfont=dict(
    #         family='Arial',
    #         size=12,
    #         color='rgb(82, 82, 82)',
    #     ),
    # ),
    #     yaxis=dict(
    #         showgrid=False,
    #         zeroline=False,
    #         showline=False,
    #         showticklabels=True,
    #         visible=True
    # ),
    #     showlegend=True,
    #     plot_bgcolor='black'
    # )
    # st.plotly_chart(fig)
# st.write('Export file for this figure is EKG_by_day.csv')
# export.to_csv('EKG_by_day.csv', index=False)
# ##########################################
elif function == 'Show an EKG':
    # selection
    ekg_df = pd.read_csv('EKGs.csv')
    ekg_df['string_pacs'] = ekg_df.PACs.astype(str)
    ekg_df.string_pacs.replace('nan', 'not able to measure', inplace=True)
    # st.write(ekg_df)
    ekg_df['show_name'] = ekg_df.name+' - '+ekg_df.string_pacs + ' PACs'
    year = st.sidebar.selectbox('Year of EKG', ['2019', '2020', '2021', '2022'], index=1)
    month = st.sidebar.selectbox(
        'Month of EKG', ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
    ekgs = ekg_df[ekg_df.name.str.contains(year+'-'+month)]
    type = st.sidebar.selectbox('Classification', list(set(ekgs.clas.tolist())))
    show_df = ekgs[ekgs.clas == type]
    show_name = st.sidebar.selectbox('Choose an EKG', show_df.show_name)
    ekg_idx = show_df[show_df.show_name == show_name].index[0]
    ekg_str = show_df.loc[ekg_idx, 'name']

# select and clean EKG to show
    ekg = Get_EKG(ekg_str, year)

    # get the classification and PAC number for this ekg from ekg_df
    this_classification = ekg_df.loc[ekg_df[ekg_df.name == ekg_str].index.tolist()[0], 'clas']
    this_PACs = ekg_df.loc[ekg_df[ekg_df.name == ekg_str].index.tolist()[0], 'PACs']

    st.write(f'You have selected {ekg_str}, classified as {this_classification}')
    ekg = Clean_EKG(ekg)

    # get singles and rate
    singles = Get_Singles(ekg)
    rate = Get_Rate(singles)
    # st.write(ekg)
    # st.write(singles)
    # plot EKG
    x = ekg.seconds
    y = ekg.micro_volts

    # set up for plotly
    PACs, level = Set_Color_For_PACs(this_PACs)
    face_color = Set_Background_Color(level)
    title = Set_Title(this_PACs, rate, PACs)

    # plotly stuff
    colorscales = px.colors.named_colorscales()
    background = px.colors.diverging.Tealrose[face_color+1]
    fig = px.line(ekg, x="seconds", y="micro_volts", width=700, height=500)
    # dotted lines for debugging
    # for i, r in singles.iterrows():
    #     fig.add_vline(x=singles.loc[i, 'seconds'], line_width=1,
    #                   line_dash="dash", line_color="green")
    fig.update_traces(line=dict(color="blue", width=0.5))
    fig.update_layout(title_text=title, title_x=0.5, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            visible=False
    ),
        showlegend=False,
        plot_bgcolor=background
    )
    st.plotly_chart(fig, use_container_width=True)
