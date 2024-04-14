import torch.nn as nn

"""
LSTM with weight tying implemented
"""


class LM_LSTM_WEIGHT_TYING(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WEIGHT_TYING, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        lstm_out, _ = self.lstm(emb)

        output = self.output(lstm_out).permute(0, 2, 1)
        return output


# Variational dropout implementation of the dropout layer
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training and not self.dropout:
            return x
        m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = m / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class LM_LSTM_VARIATIONAL_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VARIATIONAL_DROPOUT, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_vdrop = VariationalDropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        self.out_vdrop = VariationalDropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.emb_vdrop(emb)
        lstm_out, _ = self.lstm(drop1)
        drop2 = self.emb_vdrop(lstm_out)
        output = self.output(drop2).permute(0, 2, 1)
        return output
