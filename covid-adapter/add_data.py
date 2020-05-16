import pandas as pd
import re


def read_regions_data(path):
    df = pd.read_excel(path)
    df['Дата'] = df['Дата'].apply(
    lambda x:
        pd.to_datetime(str(x), format='%y-%m-%d').date()
    )
    df = df.fillna(0)
    dropout_list = ['г. Севастополь', 'Россия без Москвы', 'г.Севастополь']
    for col in dropout_list:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df['Россия без Москвы'] = df['Россия'] - df['Москва']
    df = df.fillna(0)
    return df

def add_data_to_csv (df1, path, path_output):
    df_parsed = pd.read_csv(path, sep='\t')
    df_parsed['Дата'] = df_parsed['Unnamed: 0'].apply(
        lambda x: pd.to_datetime(x).date()
    )
    df_parsed = df_parsed.drop('Unnamed: 0', axis=1)
    #df_parsed['Россия'] = df_parsed[[x for x in df_parsed.columns if x!= 'Дата']].sum(axis=0)
    
    dates = df1['Дата'].values
    
    df_new = df_parsed[
        ~df_parsed['Дата'].isin(dates)
    ]
    
    df = pd.concat([df1, df_new], sort=True, ignore_index=True, join='outer').sort_values(by='Дата').reset_index(drop=True)
    df['Дата'] = df['Дата'].apply(lambda x: x.strftime('%y-%m-%d'))
    df[['Дата'] + [col for col in df.columns if col != 'Дата']].to_excel(path_output, index=False)
    
    return 0