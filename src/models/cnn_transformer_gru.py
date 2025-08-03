# models/cnn_transformer_gru.py
import torch
import torch.nn as nn


class CnnTransformerGru(nn.Module):
    def __init__(self, in_channels, num_classes, **model_params):
        super(CnnTransformerGru, self).__init__()

        cnn_out_channels = model_params.get('cnn_out_channels', 32)
        d_model = model_params.get('d_model', 64)
        nhead = model_params.get('nhead', 4)
        num_transformer_layers = model_params.get('num_transformer_layers', 2)
        gru_hidden_size = model_params.get('gru_hidden_size', 64)
        gru_num_layers = model_params.get('gru_num_layers', 1)
        dropout = model_params.get('dropout', 0.5)

        # --- 1. CNN Encoder ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16, cnn_out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # 假设输入长度为 240 (60s*4Hz), 经过CNN后长度变为 240/16 = 15

        # --- 2. Transformer Encoder ---
        # Transformer 需要 d_model == cnn_out_channels
        if cnn_out_channels != d_model:
            self.embedding = nn.Linear(cnn_out_channels, d_model)
        else:
            self.embedding = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- 3. GRU Decoder/Aggregator ---
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # --- 4. Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, in_channels, seq_len)

        # Pass through CNN
        x = self.cnn_encoder(x)  # -> (batch, cnn_out_channels, new_seq_len)

        # Prepare for Transformer
        x = x.permute(0, 2, 1)  # -> (batch, new_seq_len, cnn_out_channels)
        x = self.embedding(x)  # -> (batch, new_seq_len, d_model)

        # Pass through Transformer
        x = self.transformer_encoder(x)  # -> (batch, new_seq_len, d_model)

        # Pass through GRU
        outputs, _ = self.gru(x)
        last_output = outputs[:, -1, :]  # -> (batch, gru_hidden_size * 2)

        # Classifier
        logits = self.classifier(last_output)

        return logits