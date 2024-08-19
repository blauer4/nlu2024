import torch
import transformers
from transformers import BertModel, BertConfig
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BertNLU(nn.Module):
    def __init__(self, out_slot, out_int, dropout=0.1, model_name="bert-base-uncased"):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.intent_out = nn.Linear(config.hidden_size, out_int)
        self.slot_out = nn.Linear(config.hidden_size, out_slot)

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_mask=attention_masks)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output_dropped = self.dropout(sequence_output)
        slots_out = self.slot_out(sequence_output_dropped)

        pooled_output_dropped = self.dropout(pooled_output)
        intent_out = self.intent_out(pooled_output_dropped)

        slots_out = slots_out.permute(0, 2, 1)  # batch_size x num_labels x seq_len
        return slots_out, intent_out
