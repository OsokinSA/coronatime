import json
import streamlit as st
import pandas as pd
import numpy as np

@st.cache(persist = True)
def read_regions_list(path):
    with open('data/regions.txt', 'r') as f:
        regions_read = json.load(f)
    return regions_read

@st.cache(persist = True)
def read_data(path):
    df = pd.read_csv(path,
                     sep=',',
                     dtype = {'region': str,
                              'population': int,
                              'density': float}
                    )
    df = df.append({'region': 'Россия без Москвы', 'population': 146745000-12678079, 'density': 8.57}, ignore_index=True)
    df = df.append({'region': 'Россия', 'population': 146745000, 'density': 5}, ignore_index=True)
    return df

@st.cache(persist = True)
def read_regions_data(path):
    df = pd.read_excel(path)
    df['Дата'] = df['Дата'].apply(
    lambda x:
        pd.to_datetime(str(x), format='%y-%m-%d')
    )
    df = df.fillna(0)
    dropout_list = ['г. Севастополь', 'Россия без Москвы']
    for col in dropout_list:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    #df['Россия без Москвы'] = df[[x for x in df.columns[1:] if x != 'Москва']].sum(axis=1)
    df['Россия без Москвы'] = df['Россия'] - df['Москва']
    df = df.fillna(0)
    #df['Россия'] = df[[x for x in df.columns[1:] if x != 'Россия без Москвы']].sum(axis=1)
    df_cum = df.copy()
    for col in df.columns[1:]:
        df_cum[col] = pd.Series(df[col]).cumsum()
    return df, df_cum

@st.cache(persist = True)
def get_sorted_regions(regions_data_cum):
    latest_entry = regions_data_cum['Дата'].max()
    cols = np.array(regions_data_cum.columns[1:])
    epid = np.array([
        regions_data_cum[regions_data_cum['Дата'] == latest_entry][col].iloc[0] for col in cols
    ])
    cols_sorted = cols[np.argsort(-1*epid)]
    return cols_sorted