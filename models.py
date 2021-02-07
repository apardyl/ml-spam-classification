import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SpamClassifier(nn.Module):
    def __init__(self, vocabulary_size, embed_size=1024, hidden_size=1024):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embed_size, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.LongTensor, lens):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        x = self.classifier(hidden[-1])
        return x
