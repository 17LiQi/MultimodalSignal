# file: model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ======================================================================================
# --- 1. Building Block: The "Expert" for a single channel ---
# ======================================================================================

class ChannelExpert(nn.Module):
    """
    An expert module that processes a single physiological signal channel.
    It consists of a series of 1D Convolutional layers followed by a GRU.
    """

    def __init__(self, input_dim=1, cnn_channels=[16, 32, 64], gru_hidden_size=64, gru_layers=1):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels

        # The output of CNNs is flattened in time and fed to GRU
        # Let's calculate the flattened size after convolutions and pooling
        # This depends on window_length, but GRU can handle variable length.
        # So we just need the feature dimension.
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True  # Bidirectional GRU often captures context better
        )

    def forward(self, x):
        # Input x shape: (Batch, Window_Length, 1) for a single channel

        # Conv1d expects (Batch, Channels, Length), so we permute
        x = x.permute(0, 2, 1)  # -> (Batch, 1, Window_Length)

        for layer in self.cnn_layers:
            x = layer(x)  # Shape changes after each block

        # After CNNs, shape is (Batch, Last_CNN_Channels, Reduced_Length)
        # GRU expects (Batch, Length, Features), so we permute again
        x = x.permute(0, 2, 1)  # -> (Batch, Reduced_Length, Last_CNN_Channels)

        # We only need the final hidden state of the GRU
        _, last_hidden_state = self.gru(x)

        # last_hidden_state shape: (num_layers * num_directions, Batch, hidden_size)
        # We concatenate the forward and backward hidden states
        # Forward is last_hidden_state[-2, :, :], Backward is last_hidden_state[-1, :, :]
        final_features = torch.cat((last_hidden_state[-2, :, :], last_hidden_state[-1, :, :]), dim=1)

        return final_features  # Shape: (Batch, 2 * gru_hidden_size)


# ======================================================================================
# --- 2. The Main Model: CrossModal Attention CNN-GRU ---
# ======================================================================================

class CrossModal_Att_CNN_GRU(nn.Module):
    """
    The main model that integrates multiple ChannelExperts with a Transformer-based
    cross-modal attention fusion mechanism.
    """

    def __init__(self, channel_names: List[str], num_classes: int, expert_config: dict, fusion_config: dict):
        super().__init__()
        self.channel_names = channel_names
        self.num_channels = len(channel_names)

        # Create an independent "expert" for each channel
        self.channel_experts = nn.ModuleList([
            ChannelExpert(**expert_config) for _ in range(self.num_channels)
        ])

        expert_output_dim = 2 * expert_config['gru_hidden_size']

        # The "Decision Committee": A Transformer Encoder layer for fusion
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=expert_output_dim,
            nhead=fusion_config['num_heads'],
            dim_feedforward=fusion_config['ff_dim'],
            dropout=fusion_config['dropout'],
            batch_first=True  # IMPORTANT!
        )
        self.fusion_layer = nn.TransformerEncoder(transformer_layer, num_layers=fusion_config['num_layers'])

        # A special token, similar to [CLS] token in BERT, to represent the aggregated state
        self.class_token = nn.Parameter(torch.zeros(1, 1, expert_output_dim))

        # The final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(expert_output_dim),
            nn.Linear(expert_output_dim, num_classes)
        )

    def forward(self, x):
        # Input x shape: (Batch, Window_Length, Total_Num_Channels)

        # 1. Split the multimodal input into individual channels
        # Shape of each channel_input: (Batch, Window_Length, 1)
        channel_inputs = torch.split(x, 1, dim=2)

        # 2. Process each channel through its dedicated expert
        expert_outputs = []
        for i, expert in enumerate(self.channel_experts):
            channel_output = expert(channel_inputs[i])  # Shape: (Batch, expert_output_dim)
            expert_outputs.append(channel_output)

        # 3. Stack the expert outputs to form a sequence for the fusion layer
        # Shape becomes (Batch, Num_Channels, expert_output_dim)
        expert_sequence = torch.stack(expert_outputs, dim=1)

        # 4. Prepend the class token to the sequence
        batch_size = x.shape[0]
        # Repeat the class token for each item in the batch
        class_token_batch = self.class_token.repeat(batch_size, 1, 1)
        # Concatenate along the sequence dimension (dim=1)
        fusion_input = torch.cat([class_token_batch, expert_sequence], dim=1)

        # 5. Pass through the Transformer fusion layer
        # The Transformer will calculate attention between all channels and the class token
        fused_representation = self.fusion_layer(fusion_input)

        # 6. Extract the output corresponding to the class token
        # This token has now aggregated information from all channels via attention
        class_token_output = fused_representation[:, 0, :]  # Shape: (Batch, expert_output_dim)

        # 7. Classify
        logits = self.classifier(class_token_output)

        return logits