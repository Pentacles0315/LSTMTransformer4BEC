# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:08:27 2024

@author: guo00
"""
from models import *
from utils import *
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def evaluate_model_X(model, dataloader, criterion, device, metrics):
    model.eval()  # 切换为评估模式
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)[:, -1, :]
            loss = criterion(outputs, batch_y.view(batch_y.shape[0], batch_y.shape[2]))
            total_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.view(batch_y.shape[0], batch_y.shape[2]).cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    performances = []
    for function in metrics:
        performance = function(all_targets, all_predictions)
        performances.append(performance)

    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, performances

def evaluate_model_XY(model, dataloader, criterion, device, metrics):
    model.eval()  # 切换为评估模式
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X, batch_y.view(batch_y.shape[0], batch_y.shape[2]))[:, -1, :]
            loss = criterion(outputs, batch_y.view(batch_y.shape[0], batch_y.shape[2]))
            total_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.view(batch_y.shape[0], batch_y.shape[2]).cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    performances = []
    for function in metrics:
        performance = function(all_targets, all_predictions)
        performances.append(performance)

    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, performances
def evaluate_model_inex(model, dataloader, criterion, device, metrics):
    model.eval()  # 切换为评估模式
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_external, batch_internal,batch_y in dataloader:
            batch_external, batch_internal,batch_y = batch_external.to(device), batch_internal.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_external, batch_internal)[:, -1, :]
            loss = criterion(outputs, batch_y.view(batch_y.shape[0], batch_y.shape[2]))
            total_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.view(batch_y.shape[0], batch_y.shape[2]).cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    performances = []
    for function in metrics:
        performance = function(all_targets, all_predictions)
        performances.append(performance)

    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, performances
def predict_X(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)[:, -1, :]

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.view(batch_y.shape[0], batch_y.shape[2]).cpu().numpy())
    return all_predictions, all_targets
def predict_XY(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X, batch_y.view(batch_y.shape[0], batch_y.shape[2]))[:, -1, :]

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.view(batch_y.shape[0], batch_y.shape[2]).cpu().numpy())
    return all_predictions, all_targets
def evaluate_Transformer(model, dataloader, criterion, device, metrics):
    model.eval()  # 切换为评估模式
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 在评估时不需要计算梯度
        for external, internal, batch_y in dataloader:
            external, internal, batch_y = external.to(device), internal.to(device), batch_y.to(device)
            
            outputs = model.predict(external, internal)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.append(outputs.reshape(-1, outputs.shape[2]).cpu().numpy())
            all_targets.append(batch_y.reshape(-1, batch_y.shape[2]).cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    performances = []
    for function in metrics:
        performance = function(all_targets, all_predictions)
        performances.append(performance)

    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, performances