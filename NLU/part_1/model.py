import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=False,
                 dropout=0.1):
        """

        :param hid_size: Hidden size
        :param out_slot: Number of slots (output size for slot filling)
        :param out_int: Number of intents (output size for intent class)
        :param emb_size: Word embedding size
        :param vocab_len: Length of the vocabulary
        :param n_layer: Number of layers
        :param pad_index: Index of the padding token
        """
        super(ModelIAS, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=self.bidirectional, batch_first=True)

        # We need to double the hidden size if the LSTM is bidirectional
        hid_size = hid_size * 2 if self.bidirectional else hid_size

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        #
        utt_emb = self.dropout(utt_emb)
        utt_emb = utt_emb.permute(1, 0, 2)  # utt_emb.size() = seq_len X batch_size X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to utt_encoded
        utt_encoded = self.dropout(utt_encoded)

        # If bidirectional, we need to concatenate the last hidden states of the forward and backward LSTMs
        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-1, :, :], last_hidden[-2, :, :]), dim=1)
        else:
            last_hidden = last_hidden[-1, :, :]

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
