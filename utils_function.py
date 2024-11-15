# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:08:39 2024

@author: guo00
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from functools import reduce

# 读取文件夹中的所有CSV文件
def load_x_csv_files(folder_path, time = 'date', format = "mixed"):
    all_data = []
    folder_path = folder_path + "/X"
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data[time] = pd.to_datetime(data[time], format=format)
            all_data.append(data)
    return all_data
    
def load_y_csv_files(folder_path, time = 'date'):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data[time] = pd.to_datetime(data[time])
    return data
def data_preprocessing(folder_path, time = 'date'):
    all_data = load_x_csv_files(folder_path, time)
    y = load_y_csv_files(folder_path, time)
    timestep = y[time]
    for data in all_data:
        timestep = pd.merge(timestep, data, on=time, how='left')
    return timestep, y
#dataloader-dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length, predict_length = 1, time = 'date'):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.predict_length = predict_length
        
        self.X = self.X.set_index(time)
        self.y = self.y.set_index(time)

        self.X = pd.merge(self.X, self.y, left_index=True, right_index=True, how='inner', suffixes=('_X', '_y'))


        # 分离出特征和标签
        self.X_data = self.X.values
        self.y_data = self.y.values
        
    def __len__(self):
        return len(self.X_data) - self.seq_length - self.predict_length + 1
    def __getitem__(self, idx):
        # 获取长度为 seq_length 的输入序列，并获取下一个时间点的目标值
        if self.predict_length == 0:
            return (
                torch.tensor(self.X_data[idx:idx + self.seq_length], dtype=torch.float32),
                torch.tensor(self.y_data[idx:idx + self.seq_length], dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(self.X_data[idx:idx + self.seq_length], dtype=torch.float32),
                torch.tensor(self.y_data[idx + self.seq_length - 1:idx + self.seq_length + self.predict_length], dtype=torch.float32)
            )

class TimeSeriesDatasetforSum(Dataset):
    def __init__(self, X, y, seq_length, predict_length = 1, time = 'date'):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.predict_length = predict_length
        
        self.X = self.X.set_index(time)
        self.y = self.y.set_index(time)

        self.X = pd.merge(self.X, self.y, left_index=True, right_index=True, how='inner', suffixes=('_X', '_y'))


        # 分离出特征和标签
        self.X_data = self.X.values
        self.y_data = pd.DataFrame(y.set_index(time).sum(axis=1)).values
        
    def __len__(self):
        if self.predict_length == 0:
            return len(self.X_data) - self.seq_length + 1
        else:
            return len(self.X_data) - self.seq_length
    
    def __getitem__(self, idx):
        # 获取长度为 seq_length 的输入序列，并获取下一个时间点的目标值
        if self.predict_length == 0:
            return (
                torch.tensor(self.X_data[idx:idx + self.seq_length], dtype=torch.float32),
                torch.tensor(self.y_data[idx:idx + self.seq_length], dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(self.X_data[idx:idx + self.seq_length], dtype=torch.float32),
                torch.tensor(self.y_data[idx + self.seq_length:idx + self.seq_length + self.predict_length], dtype=torch.float32)
            )
class TimeSeriesDataset_sep(Dataset):
    def __init__(self, X, y, seq_length, predict_length = 1, time = 'date'):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.predict_length = predict_length
        
        self.X = self.X.set_index(time)
        self.y = self.y.set_index(time)


        # 分离出特征和标签
        self.X_data = self.X.values
        self.y_data = self.y.values
        
    def __len__(self):
        return len(self.X_data) - self.seq_length - self.predict_length + 1
    
    def __getitem__(self, idx):
        # 获取长度为 seq_length 的输入序列，并获取下一个时间点的目标值
        return (
            torch.tensor(self.X_data[idx:idx + self.seq_length], dtype=torch.float32),
            torch.tensor(self.y_data[idx:idx + self.seq_length], dtype=torch.float32),
            torch.tensor(self.y_data[idx + self.seq_length:idx + self.seq_length + self.predict_length], dtype=torch.float32)
        )

#datasets
def CU_BEMS():
    dataframes_2018 = [
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor1_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor2_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor3_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor4_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor5_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor6_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2018Floor7_clean.csv")
    ]
    for i, df in enumerate(dataframes_2018, start=1):
        df.columns = df.columns.str.strip().str.lower()  # 去掉列名前后空格并转为小写
    
        # 确保 Date 列为索引，不参与预测
        df['date'] = pd.to_datetime(df['date'])
        df.columns = ['date' if col == 'date' else f'Floor{i}_{col}' for col in df.columns]

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dataframes_2018)
    merged_df.set_index('date', inplace=True)
    merged_df.columns = merged_df.columns.str.replace(r'Floor\d+_', '', regex=True)
    merged_df = merged_df.T.groupby(merged_df.columns, sort=False).sum().T
    x_columns = [col for col in merged_df.columns if '(kw)' not in col]  # 非能耗列
    y_columns = [col for col in merged_df.columns if '(kw)' in col]      # 能耗列
    
    X_2018 = merged_df[x_columns]
    y_2018 = merged_df[y_columns]
    dataframes_2019 = [
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor1_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor2_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor3_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor4_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor5_clean.csv"),
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor6_clean.csv"), 
        pd.read_csv("datasets/CU-BEMS_clean/2019Floor7_clean.csv")
    ]

    # 初始化空DataFrame

    # 处理每个楼层的数据
    for i, df in enumerate(dataframes_2019, start=1):
        df.columns = df.columns.str.strip().str.lower()  # 去掉列名前后空格并转为小写
    
        # 确保 Date 列为索引，不参与预测
        df['date'] = pd.to_datetime(df['date'])
        df.columns = ['date' if col == 'date' else f'Floor{i}_{col}' for col in df.columns]

    merged_df_2019 = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dataframes_2019)
    merged_df_2019.set_index('date', inplace=True)
    merged_df_2019.columns = merged_df_2019.columns.str.replace(r'Floor\d+_', '', regex=True)
    merged_df_2019 = merged_df_2019.T.groupby(merged_df_2019.columns, sort=False).sum().T

    x_columns_2019 = [col for col in merged_df_2019.columns if '(kw)' not in col]  # 非能耗列
    y_columns_2019 = [col for col in merged_df_2019.columns if '(kw)' in col]      # 能耗列
    
    X_2019 = merged_df_2019[x_columns_2019]
    y_2019 = merged_df_2019[y_columns_2019]

    X = pd.concat([X_2018, X_2019], axis=0)
    X.reset_index(inplace=True)
    y = pd.concat([y_2018, y_2019], axis=0)
    y["sum"] = y.sum(axis = 1)
    y.reset_index(inplace=True)
    y = y[["date", "sum"]]
    return X, y
def LBNL59(): 
    folder_path = "datasets/LBNL59"
    X,y = data_preprocessing(folder_path)
    return X, y
def occupant():
    folder_path = "datasets/occupant_data"
    X,y = data_preprocessing(folder_path, time = "timestamp [dd/mm/yyyy HH:MM]")
    columns_list = []
    for i in X.columns:
        if "ki" in i or "mr" in i:
            pass
        else:
            columns_list.append(i)
    X = X[columns_list]
    X_columns_to_normalize = X.columns.difference(['timestamp [dd/mm/yyyy HH:MM]'])
    X_scaler = MinMaxScaler()
    X[X_columns_to_normalize] = X_scaler.fit_transform(X[X_columns_to_normalize])
    y_columns_to_sum = y.columns.difference(['timestamp [dd/mm/yyyy HH:MM]'])
    y['sum'] = y[y_columns_to_sum].sum(axis=1)
    y = y[["timestamp [dd/mm/yyyy HH:MM]", 'sum']]
    return X, y