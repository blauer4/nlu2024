# Add functions or classes used for data loading and preprocessing
import os
import json
import torch
import torch.utils.data as data
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

BERT_MODEL = "bert-base-uncased"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
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
        for word in elements:
            vocab[word] = tokenizer.convert_tokens_to_ids(word)
        count = Counter(elements)
        for k, v in sorted(count.items()):
            if v > cutoff:
                vocab[k] = tokenizer.convert_tokens_to_ids(k)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab[PAD_TOKEN] = PAD_ID
        for elem in sorted(set(elements)):
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
        utt = torch.Tensor(tokenizer.build_inputs_with_special_tokens(self.utt_ids[idx]))
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


class BERTIntentSlotDataset(data.Dataset):
    def __init__(self, data, intent_label_map, slot_label_map):
        """
        Args:
            data (list of dict): The dataset, where each item is a dict with 'utterance', 'intent', and 'slots'.
            intent_label_map (dict): A mapping from intent labels to their corresponding IDs.
            slot_label_map (dict): A mapping from slot labels to their corresponding IDs.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.intent_label_map = intent_label_map
        self.slot_label_map = slot_label_map
        self.pad_token_label_id = PAD_ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the relevant data
        utterance = self.data[idx]['utterance']
        intent_label = self.intent_label_map[self.data[idx]['intent']]
        slot_labels = self.data[idx]['slots']

        tokenized = self.tokenizer(utterance, return_tensors='pt')

        input_ids = tokenized['input_ids'][0]  # Remove batch dimension
        attention_mask = tokenized['attention_mask'][0]  # Remove batch dimension

        # Get the word to token mapping and original words
        word_ids = tokenized.word_ids()

        # Adjust word_ids to account for special tokens
        delta = 0
        prev_word = None
        for i, w in enumerate(word_ids):
            if w is None or w == 0:
                continue
            char_span = tokenized.word_to_chars(w)
            if utterance[char_span[0] - 1] != ' ' and prev_word != w:
                # check if there is a space before the word and the word before has it
                delta += 1
            prev_word = w
            word_ids[i] = w - delta

        # Prepare slot labels (align them with tokens)
        aligned_labels = []
        last_word_id = None
        slot_labels_list = slot_labels.split()

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # For special tokens, set the label to pad_token_label_id
                aligned_labels.append(self.pad_token_label_id)
            elif word_id != last_word_id:
                # Only the first subtoken of a word gets the label
                label = slot_labels_list[word_id]
                aligned_labels.append(self.slot_label_map[label])
            else:
                # Subsequent subtokens get the padding label
                aligned_labels.append(self.pad_token_label_id)
            last_word_id = word_id

        # Convert to tensor
        slot_label_ids = torch.tensor(aligned_labels, dtype=torch.long)
        intent_label = torch.tensor(intent_label, dtype=torch.long)
        sentence = utterance.split()
        if len(sentence) != (max(word_ids[1:-1]) + 1):
            print("ERROR")
            print(sentence)
            print(word_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'intent_label': intent_label,
            'slot_labels': slot_label_ids,
            'sentence': sentence
        }


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # So we create a matrix full of PAD_ID with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_ID)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['input_ids']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['input_ids'])
    y_slots, y_lengths = merge(new_item["slot_labels"])
    attention_mask, _ = merge(new_item["attention_mask"])
    intent = torch.LongTensor(new_item["intent_label"])

    attention_mask = attention_mask.to(DEVICE)
    src_utt = src_utt.to(DEVICE)  # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    new_item['attention_mask'] = attention_mask
    new_item["input_ids"] = src_utt
    new_item["intent_labels"] = intent
    new_item["slot_labels"] = y_slots
    new_item["slot_len"] = y_lengths
    return new_item

