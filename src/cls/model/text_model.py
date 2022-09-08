import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, n_vocabs, channels=512, kernel_size=5, depth=3, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(n_vocabs, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ))
        self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        self.ln = nn.Sequential(nn.Linear(channels * 2, channels))

    def forward(self, x, attention_mask):
        input_lengths = attention_mask.sum(-1)
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        x = self.cnn(x)  # [B, chn, T]
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, (h, c) = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        return x.sum(dim=1)

