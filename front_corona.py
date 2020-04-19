import streamlit as st
import pandas as pd
import numpy as np
import datetime
from get_data import read_regions_list, read_data, read_regions_data, get_sorted_regions
from plot_data import plot_line, add_explanation
from show_stats import show_latest_stats

#import altair as alt
#import pydeck as pdk

date_max = datetime.date(2020,4,1)
date_min = datetime.date(2020,5,1)

st.title("COVID-19")

regions_data, regions_data_cum = read_regions_data('data/regions_data.xlsx')
regions_population = read_data('data/regions_population.csv')
regions_list = get_sorted_regions(regions_data_cum)

regions = st.sidebar.multiselect(
    'Выберите регион', list(regions_list)
)

log_scale = st.sidebar.checkbox(
    'Логарфим. шкала'
)

mode = st.sidebar.selectbox(
    'Метод прогноза', ['exp', 'epid', 'polynom_3']
)

date_0 = st.sidebar.date_input('Отображать ОТ', datetime.date(2020,4,1))
date_1 = st.sidebar.date_input('Отображать ДО', datetime.date(2020,5,1))

date_pred_0 = st.sidebar.date_input('Отрезок для прогноза ОТ', datetime.date(2020,4,2))
date_pr = regions_data['Дата'].max() - np.timedelta64(3, 'D')
#date_pred_1 = st.sidebar.date_input('Отрезок для прогноза ДО', datetime.date(2020,4,10))
date_pred_1 = st.sidebar.date_input('Отрезок для прогноза ДО', date_pr)

#st.write('Вы выбрали регионы: ' + ' '.join(regions))
warning = 0
if date_pred_0 > date_pred_1:
    st.warning('Неверно указан отрезок для прогноза')
    warning = 1
if date_pred_0 < date_0:
    st.warning('Начало отрезка для прогноза должно содержаться в отрезке для отображения')
    warning = 1
if (date_1 <= date_0) | (date_pred_1 <= date_pred_0):
    st.warning('"Я им сначала покажу конец, а в конце начало покажу, потом все скажут что я гений." К. Нолан')
    warning = 1
if len(regions)>5:
    st.warning('Слишком много регионов выбрано. Все на график не влезут.')
    warning = 1
if len(regions)==0:
    st.warning('Для начала выберите регион.')
    warning = 1
else:
    show_latest_stats(regions_data, regions_data_cum, regions_population, regions)

dates = (np.datetime64(date_0), np.datetime64(date_1), np.datetime64(date_pred_0), np.datetime64(date_pred_1))

if warning == 0:
    plot_line(regions_data_cum, regions_data, regions, regions_population, log_scale, dates, mode, 'Суммарное кол-во случаев заражений')
    plot_line(regions_data_cum, regions_data, regions, regions_population, log_scale, dates, mode+'_delta', 'Прирост за день')

if (len(regions)!=0) & (warning==0):
    add_explanation(mode)

st.sidebar.markdown('Источник данных: https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_COVID-19_%D0%B2_%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D0%B8')
st.sidebar.markdown('Все прогнозы вымышлены и скорее всего не сбудутся, особенно долгосрочные.')
st.sidebar.markdown('Сидите лучше по домам!')
st.sidebar.markdown('ЗделОл: Осокин Сергей')