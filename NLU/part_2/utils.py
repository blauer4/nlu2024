# Add functions or classes used for data loading and preprocessing
import os
import json
import torch
import torch.utils.data as data
from collections import Counter
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
PAD_TOKEN = tokenizer.pad_token
PAD_ID = tokenizer.pad_token_id
UNK_TOKEN = tokenizer.unk_token
UNK_ID = tokenizer.unk_token_id
SEP_TOKEN = tokenizer.sep_token
SEP_ID = tokenizer.sep_token_id
CLS_TOKEN = tokenizer.cls_token
CLS_ID = tokenizer.cls_token_id


def load_data(path):
    """
        input: path/to/data
        output: json
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def divide_training_set(tmp_train_raw, test_raw):
    """
    Divide the training set into training and validation set
    :param tmp_train_raw: The full training set loaded from the json file
    :param test_raw: The full test set loaded from the json file
    """
    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True,
                                                      stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    val_raw = X_val

    y_test = [x['intent'] for x in test_raw]
    return train_raw, val_raw, y_train, y_val, y_test


"""
Other codeblock
"""


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {PAD_TOKEN: PAD_ID, CLS_TOKEN: CLS_ID, SEP_TOKEN: SEP_ID}
        if unk:
            vocab[UNK_TOKEN] = UNK_ID
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab[PAD_TOKEN] = PAD_ID
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk=UNK_TOKEN):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        attention_masks = torch.FloatTensor([[1 for i in range(len(seq))] + [0 for i in range(max_len - len(seq))]for seq in sequences])
        # So we create a matrix full of PAD_ID with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_ID)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths, attention_masks

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _, attention_masks = merge(new_item['utterance'])
    y_slots, y_lengths, _ = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    attention_masks = attention_masks.to(DEVICE)
    src_utt = src_utt.to(DEVICE)  # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    new_item['attention_masks'] = attention_masks
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item
