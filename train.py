# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:56:41 2024

@author: guo00
"""

from models import *
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 8 #input_size is count of features
hidden_size = 64
num_layers = 4
output_size = 10  # predicted time step

model = LSTM(input_size, hidden_size, num_layers, output_size)


def train(model, train_loader, num_epochs, learning_rate, device):
     
    model.to(device)
    
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return model
