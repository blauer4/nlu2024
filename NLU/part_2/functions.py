import copy

import numpy as np
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from conll import evaluate
from model import *
from utils import *


def run_experiments(to_run):
    save_path = "./"
    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))
    train_raw, val_raw, y_train, y_val, y_test = divide_training_set(tmp_train_raw, test_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + val_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    default_options = {
        'lr': 0.0001,
        'clip': 5,
        'dropout': 0,
        'epochs': 30,
        'patience': 10,
        'run': False,
    }

    for experiment in to_run:
        arg = default_options | to_run[experiment]
        print(f"Running experiment {experiment}")

        if arg['run']:
            lang = Lang(words, intents, slots, cutoff=0)
        else:
            saved_model = torch.load('./bin/' + experiment + '.pt', map_location=torch.device(DEVICE))
            lang = saved_model['lang']

        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)

        # train_dataset = IntentsAndSlots(train_raw, lang)
        train_dataset = BERTIntentSlotDataset(train_raw, lang.intent2id, lang.slot2id)
        # val_dataset = IntentsAndSlots(val_raw, lang)
        val_dataset = BERTIntentSlotDataset(val_raw, lang.intent2id, lang.slot2id)
        # test_dataset = IntentsAndSlots(test_raw, lang)
        test_dataset = BERTIntentSlotDataset(test_raw, lang.intent2id, lang.slot2id)

        # Dataloader instantiations
        train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

        d = datetime.now()
        strftime = d.strftime("%Y-%m-%d_%H-%M")
        runpath = save_path + 'runs/' + experiment + '/' + strftime + '/'
        os.makedirs(runpath, exist_ok=True)
        os.makedirs('./bin', exist_ok=True)
        file_path = './bin/' + experiment + '.pt'

        model = BertNLU(out_slot, out_int, dropout=arg['dropout'], model_name=BERT_MODEL).to(DEVICE)
        if to_run[experiment]['run']:
            writer = SummaryWriter(log_dir=f"{save_path}runs/{experiment}/{strftime}/")
            results_test, intent_test = train((writer, file_path), lang, model, PAD_ID, train_loader, val_loader,
                                              test_loader, arg['lr'], arg['clip'], arg['epochs'], arg['patience'])
        else:
            model.load_state_dict(saved_model['model'])
            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            criterion_intents = nn.CrossEntropyLoss()
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model,
                                                     lang.id2intent, lang.id2slot)

        if to_run[experiment]['run']:
            f = open(runpath + 'results.txt', "a")

        print(f"Slot F1: {results_test['total']['f']:.3f}")
        print(f"Intent Accuracy: {intent_test['accuracy']:.3f}")
        if to_run[experiment]['run']:
            f.write(f"Slot F1: {results_test['total']['f']:.3f}\nIntent Accuracy: {intent_test['accuracy']:.3f}\n")

        if to_run[experiment]['run']:
            f.close()


def log_values(writer, step, loss, prefix, f1_score=None, intent_acc=None):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    if f1_score is not None:
        writer.add_scalar(f"{prefix}/f1_score", f1_score, step)
    if intent_acc is not None:
        writer.add_scalar(f"{prefix}/intent_acc", intent_acc, step)


def train(logging, lang, model, PAD_ID, train_loader, val_loader, test_loader, lr, clip, epochs=10, pat=3):
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_f1 = 0
    best_model = None
    writer, file_path = logging
    patience = pat

    pbar = tqdm(range(1, epochs))

    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        log_values(writer, x, np.asarray(loss).mean(), 'train')

        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_val, intent_res, loss_val = eval_loop(val_loader, criterion_slots, criterion_intents, model,
                                                      lang.id2intent, lang.id2slot)
        losses_val.append(np.asarray(loss_val).mean())

        f1 = results_val['total']['f']
        log_values(writer, x, np.asarray(loss_val).mean(), 'val', f1, intent_res['accuracy'])
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        print(f1, intent_res['accuracy'])
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model).to('cpu')
            patience = pat
        else:
            patience -= 1
        if patience <= 0:  # Early stopping with patience
            break  # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    to_save = {
        "model": best_model.state_dict(),
        "lang": lang,
    }
    torch.save(to_save, file_path)
    writer.close()
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model,
                                             lang.id2intent, lang.id2slot)
    return results_test, intent_test


def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()  # Set the model to training mode
    loss_array = []

    for sample in data_loader:
        optimizer.zero_grad()
        slots_logits, intents_logits = model(sample["input_ids"], sample["attention_mask"])

        loss_intent = criterion_intents(intents_logits, sample['intent_labels'])
        loss_slot = criterion_slots(slots_logits, sample['slot_labels'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_array


def eval_loop(data_loader, criterion_slots, criterion_intents, model, intent_label_map, slot_label_map):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    id2intent = intent_label_map
    id2slot = slot_label_map

    with torch.no_grad():  # Avoid the creation of computational graph
        for sample in data_loader:
            # Forward pass: Get model predictions
            slots_logits, intents_logits = model(sample["input_ids"], sample["attention_mask"])

            # Calculate losses
            loss_intent = criterion_intents(intents_logits, sample['intent_labels'])
            loss_slot = criterion_slots(slots_logits, sample['slot_labels'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            predicted_intents = torch.argmax(intents_logits, dim=1).tolist()
            gt_intents = sample['intent_labels'].tolist()
            ref_intents.extend([id2intent[x] for x in gt_intents])
            hyp_intents.extend([id2intent[x] for x in predicted_intents])
            # Slot inference
            output_slots = torch.argmax(slots_logits, dim=1)
            for id_seq, seq in enumerate(output_slots):

                pred_slot_ids = output_slots[id_seq].tolist()
                remove_index = []
                gt_slot_ids = sample['slot_labels'][id_seq].tolist()
                for i_el, token in enumerate(gt_slot_ids):
                    if token == PAD_ID:
                        remove_index.append(i_el)
                pred_slot_ids = [token for id_el, token in enumerate(pred_slot_ids) if id_el not in remove_index]
                gt_slot_ids = [token for id_el, token in enumerate(gt_slot_ids) if id_el not in remove_index]

                gt_slots = [id2slot[elem] for elem in gt_slot_ids]
                pred_slots = [id2slot[elem] for elem in pred_slot_ids]

                utterance = sample["sentence"][id_seq]

                if len(utterance) != len(pred_slot_ids): breakpoint()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                hyp_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(pred_slots)])
    # Calculate metrics
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    # Generate classification report for intents
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

    return results, report_intent, loss_array
