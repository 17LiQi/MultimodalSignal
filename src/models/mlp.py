# models/mlp.py
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, **model_args):
        super(MLP, self).__init__()

        hidden_layers = model_args.get('hidden_layers', [128, 64])
        dropout = model_args.get('dropout', 0.5)

        layers = []
        input_dim = in_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x 初始形状: (batch_size, in_features)
        x = self.network(x)
        x = self.classifier(x)
        return x