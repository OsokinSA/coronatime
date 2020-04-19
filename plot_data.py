import streamlit as st
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from datetime import timedelta

def get_prediction(df, dates, cols):
    a,b,c,d = get_prediction_exp(df[(df['Дата'] >= dates[2]) & (df['Дата'] <= dates[3])],
                                 cols,'exp')                       

def get_label(mode, popt, r2):
    r2_l = ' $R^2=${:.4f} '.format(r2)
    if mode == 'exp':
        sign =  '+' if popt[0] > 0 else ''
        label_1 = '{:.2f}(x'.format(popt[1])
        label_2 = '{:.1f})'.format(popt[0])
        return r2_l + ' $e^{'+label_1+sign+label_2+'}$'
    elif mode == 'exp_delta':
        sign =  '+' if popt[0] > 0 else ''
        label_1 = '{:.2f}(x'.format(popt[1])
        label_2 = '{:.1f})'.format(popt[0])
        return r2_l + ' $\Delta e^{'+label_1+sign+label_2+'}$'
    elif mode == 'epid':
        return r2_l
    elif mode == 'epid_delta':
        return r2_l
    elif mode == 'polynom_3':
        return r2_l
    elif mode == 'polynom_3_delta':
        return r2_l
    else:
        return ''
    
def plot_line(data, data_delta, cols, regions_population, scale, dates, mode, title):
   
    delta = (dates[1] - dates[0]).astype('timedelta64[D]')
    date_pr = []
    for i in range(delta.astype('int') + 1):
        date_pr.append(np.datetime64(dates[0]) + np.timedelta64(i, 'D')) 
        
    if '_delta' in mode:
        df = data_delta[(data_delta['Дата'] >= dates[0]) & (data_delta['Дата'] <= dates[1])]
        df_pr = data_delta[(data_delta['Дата'] >= dates[2]) & (data_delta['Дата'] <= dates[3])]
        values_prediction, popt_list, r2 = get_prediction_exp(data, data_delta, regions_population, dates, cols, mode)
    else:
        df = data[(data['Дата'] >= dates[0]) & (data['Дата'] <= dates[1])]
        df_pr = data[(data['Дата'] >= dates[2]) & (data['Дата'] <= dates[3])]
        values_prediction, popt_list, r2 = get_prediction_exp(data, data_delta, regions_population, dates, cols, mode)
        
    if not scale:
#        st.line_chart(
#            df.rename(columns={'date':'index'}).set_index('index'),
#            )
        for ind, col in enumerate(cols):
            plt.plot(df['Дата'], df[col], '--o', label=col + get_label(mode, popt_list[ind], r2[ind]))
            plt.plot(df_pr['Дата'], df_pr[col], '-k', alpha=0.3, linewidth=5)
            plt.plot(date_pr, values_prediction[ind], '-r', alpha=0.3, linewidth=5)
            if r2[ind] is None:
                st.warning('для региона "' + col + '" не удалось построить прогноз')
    else:
        for ind, col in enumerate(cols):
            plt.semilogy(df['Дата'], df[col], '--o', label=col + get_label(mode, popt_list[ind], r2[ind]))
            plt.semilogy(df_pr['Дата'], df_pr[col], '-k', alpha=0.3, linewidth=5)
            plt.semilogy(date_pr, values_prediction[ind], '-r', alpha=0.3, linewidth=5)
            if pd.isnull(r2[ind]):
                st.warning('для региона "' + col + '" не удалось построить прогноз')
    plt.grid(which ='both')
    plt.xticks(rotation='vertical')
    plt.legend(loc='upper left')
    plt.title(title)
    st.pyplot()
    return 0

def exp_func(x, b, c):
    return np.exp(c*(x-b))
def exp_func_delta(x, b, c):
#   return c*np.exp(c*(x-b))
    return np.exp(c*(x-b)) - np.exp(c*(x-1-b))
def epid_func(x, g, b, N, I0):
    Ii = (1-g/b)*N*1000000
    V = Ii/(I0*1000000) - 1
    xi = b - g
    return Ii/(1+V*np.exp(-xi*x))
def epid_func_delta(x, g, b, N, I0):
    Ii = (1-g/b)*N*1000000
    V = Ii/(I0*1000000) - 1
    xi = b - g
    return epid_func(x, g, b, N, I0) - epid_func(x-1, g, b, N, I0)
#def polinom_2_finc(x, a, b, c):
#    return a(x**2) + b*(x) + c
def polynom_3_func(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d
def polynom_3_func_delta(x, a, b, c, d):
    return polynom_3_func(x,a,b,c,d) - polynom_3_func(x-1,a,b,c,d) 


def fit_exp(x,y):
    popt, pcov = curve_fit(exp_func, x, y, p0 = [0, 0.2], bounds=((-100, -100), (100, 10)))
    return popt, pcov
def fit_exp_delta(x,y):
    popt, pcov = curve_fit(exp_func_delta, x, y, p0 = [0, 0.2], bounds=((-100, -100), (100, 10)))
    return popt, pcov
def fit_epid(x,y, popul):
    popt, pcov = curve_fit(lambda x,g,b: epid_func(x,g,b,popul, y[0]/1000000), x, y, p0 = [0.01, 0.10], bounds=((0.001, 0.01), (0.8, 0.9)), method='trf')
    return popt, pcov
def fit_epid_delta(x,y, popul):
    popt, pcov = curve_fit(lambda x,g,b: epid_func_delta(x,g,b,popul, y[0]/1000000), x, y, p0 = [0.01, 0.10], bounds=((0.001, 0.01), (0.8, 0.9)), method='trf')
    return popt, pcov
def fit_3_poly(x,y):
    p = np.polyfit(x, y, 3)
    return p
#def fit_2_poly(x,y,2):
#    p, _ = polyfit(x, y, 3)
    


def dates_to_diffs(dates_0, dates_1):
    x_min = dates_0.min()
    return np.array([(val - x_min).astype('timedelta64[D]').astype('int') for val in dates_0]), np.array([(val - x_min).astype('timedelta64[D]').astype('int') for val in dates_1])


def get_prediction_exp(df, df_delta, regions_population, dates, cols, mode):
    delta = (dates[1] - dates[0]).astype('timedelta64[D]')
    date_pr = []
    for i in range(delta.astype('int') + 1):
        date_pr.append(np.datetime64(dates[0]) + np.timedelta64(i, 'D'))
    x_predict, x_fit  = dates_to_diffs(
        np.array(date_pr),
        df[(df['Дата']>=dates[2])&(df['Дата']<=dates[3])]['Дата'].values
    )
    np_list = []
    popt_list = []
    r2_list = []
    for col in cols:
        y_data = df[(df['Дата']>=dates[2]) & (df['Дата']<=dates[3])][col].values
        if mode == 'exp':
            popt, pcov = fit_exp(x_fit, y_data)
            y_predict = exp_func(x_predict, popt[0], popt[1])
            y_fit = exp_func(x_fit, popt[0], popt[1])
        elif mode == 'exp_delta':
#            popt, pcov = fit_exp(x_fit, y_data)
            popt, pcov = fit_exp(x_fit, y_data)
            y_predict = exp_func_delta(x_predict, popt[0], popt[1])
            y_fit = exp_func_delta(x_fit, popt[0], popt[1])
        elif mode == 'epid':
            popul = regions_population[regions_population['region'] == col]['population'].iloc[0]/1000000
            popt, pcov = fit_epid(x_fit, y_data, popul)
            y_predict = epid_func(x_predict, popt[0], popt[1], popul, y_data[0]/1000000)            
            y_fit = epid_func(x_fit, popt[0], popt[1], popul, y_data[0]/1000000)            
        elif mode == 'epid_delta':
            popul = regions_population[regions_population['region'] == col]['population'].iloc[0]/1000000
            popt, pcov = fit_epid(x_fit, y_data, popul)
            y_predict = epid_func_delta(x_predict, popt[0], popt[1], popul, y_data[0]/1000000)
            y_fit = epid_func_delta(x_fit, popt[0], popt[1], popul, y_data[0]/1000000)
        elif mode == 'polynom_3':
            popt = fit_3_poly(x_fit, y_data)
            y_predict = polynom_3_func(x_predict, popt[0], popt[1], popt[2], popt[3])
            y_fit = polynom_3_func(x_fit, popt[0], popt[1], popt[2], popt[3])
        elif mode == 'polynom_3_delta':
            popt = fit_3_poly(x_fit, y_data)
            y_predict = polynom_3_func_delta(x_predict, popt[0], popt[1], popt[2], popt[3])
            y_fit = polynom_3_func_delta(x_fit, popt[0], popt[1], popt[2], popt[3])
        r2_list.append(np.corrcoef(y_data, y_fit)[0,1]**2)
        np_list.append(y_predict)
        popt_list.append(popt)
    return np_list, popt_list, r2_list

def add_explanation(mode):
    if mode == 'exp':
        st.markdown('Приближение по формуле: $I(t) = e^{k(t-d)}$')
        st.markdown('Не годится для долгосрочного прогноза')
    elif mode == 'epid':        
        st.markdown('Приближение по формуле: $I(t) = \dfrac{I_{\infty}}{1+ (I_{\infty}/I_{0} - 1) e^{-(\gamma-b)t}}$')
        st.markdown('Предназначена формула для долгосрочного прогноза. Но мне совсем не нравится прогноз данных, он не похож на исторические данные, напрмер для Ухани. Я ещё попробую подогнать параметры для подбора решения.')
    elif mode == 'polynom_3':
        st.markdown('Приближение по формуле: $I(t) = at^{3}+bt^{2}+ct+d$')
        st.markdown('Работает если данных совсем мало, но я не знаю зачем тогда вам прогноз на атаких данных. Если Вам всё равно и вы не боитесь богов, то используйте на здоровье.')