import torch.nn as nn


class LM_RNN(nn.Module):
    """
    A simple RNN model
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0, emb_dropout=0, n_layers=1):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0, emb_dropout=0, n_layers=1):
        """
        LSTM Model with dropout implemented when embed and output probability is > 0
        :param emb_size: the size of the embedding
        :param hidden_size: The size of the hidden layer, which is of type LSTM
        :param output_size: The output size of the model
        :param pad_index: Which padding index to use
        :param out_dropout: The dropout probability for the output layer
        :param emb_dropout: The dropout probability for the embedding layer
        :param n_layers: The number of hidden layers we want for the LSTM
        """
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.drop = nn.Identity() if emb_dropout <= 0 else nn.Dropout(p=emb_dropout)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        self.drop2 = nn.Identity() if out_dropout <= 0 else nn.Dropout(p=out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop = self.drop(emb)
        lstm_out, _ = self.lstm(drop)
        drop2 = self.drop2(lstm_out)
        output = self.output(drop2).permute(0, 2, 1)
        return output
