# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:56:09 2024

@author: guo00
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义BiLSTM层，注意设置 bidirectional=True
        self.lstm1 = nn.LSTM(input_size, hidden_size*2, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*4, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # 输出层，输出的大小为 future_steps * output_size
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        # BiLSTM 前向传播
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        return out
        

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        # 定义BiLSTM层，注意设置 bidirectional=True
        
        # Transformer Encoder Layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        # Fully connected layer for multi-step and multi-variable prediction
        self.output_projection = nn.Linear(input_size, input_size)

        self.bilstm = nn.LSTM(input_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    def positional_encoding(self, seq_length, d_model, device):
        pe = torch.zeros(seq_length, d_model).to(device)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_length, d_model)
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        _, seq_length, input_size = x.size()
        
        # 生成并添加位置编码
        pe = self.positional_encoding(seq_length, input_size, x.device)
        x = x + pe  # 将位置编码加到输入上
                
        # Transpose for transformer input format
        transformer_out = self.output_projection(self.transformer_encoder(x))
        
        residual_out = torch.cat([x, transformer_out], dim=2)

        residual_out = self.leakyrelu(residual_out)
        
        bilstm_out, _ = self.bilstm(residual_out)
        
        bilstm_out = self.leakyrelu(bilstm_out)

        bilstm_out = self.leakyrelu(bilstm_out)

        out = self.fc(bilstm_out)
        # Reshape to have (batch_size, num_steps, num_variables)
        
        return out


class BiLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads, predict_length, dropout=0.1):
        super(BiLSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # BiLSTM Layer
        self.bilstmencoder = BiLSTMEncoder(hidden_size, hidden_size*2, num_layers, hidden_size, num_heads, dropout=0.1)
        self.relu = nn.ReLU()
        
        # Transformer Decoder Layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers*2)

        self.tgt_projection = nn.Linear(output_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        self.pred_length = predict_length
    def forward(self, external, internal, y):
        X = self.input_projection(torch.cat([external, internal], dim=2))
        
        y = self.tgt_projection(y)

        encoder_output = self.relu(self.bilstmencoder(X))
        
        output = self.output(self.relu(self.transformer_decoder(y, encoder_output)))
        
        return output
    def predict(self, external, internal):
        src = self.input_projection(torch.cat([external, internal], dim=2))
        batch_size, seq_leng, feat = src.shape
        memory = self.bilstmencoder(src)
        tgt = internal[:, -1:, :]
        predictions = []
        for _ in range(self.pred_length):
            output = self.output(self.transformer_decoder(self.tgt_projection(tgt), memory))
            predictions.append(output[:, -1:, :])
            tgt = torch.cat([tgt, output[:, -1:, :]], dim=1)
        return torch.cat(predictions, dim=1)
