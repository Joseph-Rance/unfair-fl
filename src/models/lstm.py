"""Implementation of LSTM"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(
        self,
        num_words: int = 30_000,
        embedding_size: int = 50,
        hidden_state_size: int = 50,
        num_layers: int = 2,
        tie_embeddings: bool = True,
        dropout: float = 0.5
    ) -> None:
        super(LSTM, self).__init__()
        self.dropout = dropout

        self.encoder = nn.Embedding(num_words, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_state_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_state_size, num_words)

        if tie_embeddings:
            assert hidden_state_size == embedding_size
            self.decoder.weight = self.encoder.weight

    def forward(self, x: torch.float) -> torch.float:

        # here we assume x is the whole sequence, so it has shape:
        #     (batch size, sequence length, embedding_size)

        out = self.encoder(x)
        out = F.dropout(out, p=self.dropout)
        out = self.lstm(out)[0]
        out = self.decoder(out)
        return out[:, -1]  # returns only the most recent output
