import torch.nn as nn


# Variational dropout implementation of the dropout layer
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training and not self.dropout:
            return x
        m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = m.detach() / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, weight_tying=True, variational_dropout=False):
        """
        LSTM model with weight tying and variational dropout implemented
        :param emb_size:
        :param hidden_size:
        :param output_size:
        :param pad_index:
        :param out_dropout:
        :param emb_dropout:
        :param n_layers:
        :param weight_tying:
        :param variational_dropout:
        """

        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.emb_vdrop = VariationalDropout(emb_dropout) if variational_dropout else nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        self.out_vdrop = VariationalDropout(out_dropout) if variational_dropout else nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        if weight_tying:
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop = self.emb_vdrop(emb)
        lstm_out, _ = self.lstm(drop)

        drop2 = self.emb_vdrop(lstm_out)

        output = self.output(drop2).permute(0, 2, 1)
        return output
