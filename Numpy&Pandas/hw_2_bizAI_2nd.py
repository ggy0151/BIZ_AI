# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import pickle
import os


data_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB'
os.chdir(data_file_path)

with open('hw_2_pickle_data_col.pkl', 'rb') as f:
    data_col = pickle.load(f)
    
GENDER_col_list, DAY_col_list, TIME_col_list, ACT_col_list = data_col

column_list = ['CUS_ID'] + GENDER_col_list + DAY_col_list + TIME_col_list + ACT_col_list


def data_preprocessing(raw_data_file, label_data_file, save_data_file):
    raw_data = pd.read_csv(raw_data_file, encoding='cp949')
    label_data = pd.read_csv(label_data_file)
    raw_data.shape
    
       
    save_df = pd.DataFrame(columns=column_list)
    first_df = pd.DataFrame(raw_data)
    
    first_df['TIME_Copy'] = pd.to_datetime(first_df['TIME_ID'], format = '%Y%m%d%H')
    first_df['hoursNum'] = first_df['TIME_Copy'].dt.hour
    first_df['TIME_HOUR'] = [data_col[2][xx] for xx in first_df['hoursNum']]
    first_df['daysNum'] = first_df['TIME_Copy'].dt.weekday
    first_df['DAY'] = [data_col[1][tt] for tt in first_df['daysNum']]
    del first_df['TIME_Copy'], first_df['hoursNum'], first_df['daysNum']

    allCUS = first_df.groupby('CUS_ID').size()
    DayFreq = first_df.groupby(['CUS_ID', 'DAY']).size()
    DayFreq = (DayFreq/allCUS).unstack()
    HourFreq = first_df.groupby(['CUS_ID', 'TIME_HOUR']).size()
    HourFreq = (HourFreq/allCUS).unstack()
    TypeFreq = first_df.groupby(['CUS_ID', 'ACT_NM']).size()
    TypeFreq = (TypeFreq/allCUS).unstack()

    df_allFreq = pd.concat([DayFreq, HourFreq, TypeFreq], sort = False).sort_index(axis=0, level=0) 
    df_allFreq = df_allFreq.groupby(level=0).sum()
    df_gender = label_data[['CUS_ID','GENDER']]
    df_gender.index = df_gender['CUS_ID']
    del df_gender['CUS_ID']
    
    allFreq = pd.concat([df_allFreq, df_gender], axis = 1, join = 'outer')
    allFreq['CUS_ID'] = allFreq.index
    allFreq = allFreq.reset_index(drop=True)
    save_df = pd.DataFrame(allFreq, columns=column_list)
    
    save_df.to_csv(save_data_file, header=True, index = False)
    

if __name__ == "__main__":
    data_preprocessing('hw_2_raw_data.csv', 'hw_2_label_data.csv', 'hw_2_data_KJY.csv')
