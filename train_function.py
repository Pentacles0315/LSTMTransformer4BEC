# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:56:41 2024

@author: guo00
"""

from models import *
from utils_function import *
import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

folder_path = "datasets/LBNL59"
X,y = data_preprocessing(folder_path)

X_columns_to_normalize = X.columns.difference(['date'])
X_scaler = MinMaxScaler()
X[y_columns_to_normalize] = X_scaler.fit_transform(self.X[X_columns_to_normalize])

seq_length = 10

dataset = TimeSeriesDataset(X, y, seq_length)
dataloader = DataLoader(dataset, batch_size=20, shuffle=False)


hidden_size = 256
num_layers = 20
num_epochs = 100
num_heads = 16

input_size = 310
output_size = 5

#model = BiLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model = BiLSTMEncoder(input_size, hidden_size, num_layers, output_size, num_heads).to(device)
criterion = nn.SmoothL1Loss()  # 用于回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

loss_per_epoch = []
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 将数据加载到 GPU 或 CPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.view(batch_y.shape[0], batch_y.shape[2]))

            # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    loss_per_epoch.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
file_name = f"model_state/{type(model).__name__}.pth"
torch.save(model.state_dict(), file_name)
