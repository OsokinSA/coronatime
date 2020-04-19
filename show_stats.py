import pandas as pd
import streamlit as st
import datetime


def show_latest_stats(regions_data, regions_data_cum, regions_population, regions):
    not_null = regions_data
    today = not_null['Дата'].max()
    
    today_new = regions_data[
        regions_data['Дата'] == today
    ].iloc[0]
    today_cum = regions_data_cum[
#        regions_data_cum['Дата'] == pd.datetime.today().date()
        regions_data_cum['Дата'] == today        
    ].iloc[0]
    st.write('По данным за %s: ' % datetime.datetime.strftime(today, format='%d-%m-%Y'))
    for region in regions:
        st.write('%s: Всего случаев заражений %i, +%i за день' % 
             (
                 region,
                 today_cum[region], today_new[region]
             )
    )