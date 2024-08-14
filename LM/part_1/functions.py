import torch.optim as optim
import numpy as np
import math
import copy
import os
from tqdm import tqdm
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from model import *
from main import DEVICE


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def run_experiments(to_run):
    save_path = "./"
    train_raw = read_file(save_path + "dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file(save_path + "dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file(save_path + "dataset/PennTreeBank/ptb.test.txt")
    # Vocab is computed only on training set
    # We add two special tokens end of sentence and padding
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=256,
                              collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=1024,
                            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024,
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    default_options = {
        'model': LM_LSTM,
        'hid_size': 200,
        'emb_size': 300,
        'lr': 2.3,
        'clip': 5,
        'emb_dropout': 0,
        'out_dropout': 0,
    }

    for experiment in to_run:
        arg = default_options | to_run[experiment]
        if to_run[experiment]['run']:
            model = arg['model'](arg['emb_size'], arg['hid_size'], vocab_len, pad_index=lang.word2id["<pad>"],
                                 out_dropout=arg['out_dropout'], emb_dropout=arg['emb_dropout']).to(DEVICE)
            model.apply(init_weights)
            optimizer = arg['optimizer'](model.parameters(), lr=arg['lr'])
            main_exp(save_path, experiment, model, optimizer, arg['clip'], train_loader, val_loader, test_loader, lang)
        else:
            model = arg['model'](arg['emb_size'], arg['hid_size'], vocab_len, pad_index=lang.word2id["<pad>"],
                                 out_dropout=arg['out_dropout'], emb_dropout=arg['emb_dropout']).to(DEVICE)
            model.load_state_dict(torch.load('./bin/' + experiment + '.pt'))
            optimizer = arg['optimizer'](model.parameters(), lr=arg['lr'])
            eval_criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
            test_ppl, _ = eval_loop(test_loader, eval_criterion, model, optimizer)
            print(f'Test ppl: {test_ppl:.2f}')


def log_values(writer, step, loss, perplexity, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/perplexity", perplexity, step)


def train_loop(data, optimizer, criterion, model, clip=5, scheduler=None):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    # Use scheduler if present to reduce lr at each epoch
    if scheduler:
        scheduler.step()
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def eval_loop(data, eval_criterion, model, optimizer):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    temp = {}
    if 't0' in optimizer.param_groups[0]:
        for prm in model.parameters():
            temp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm].clone()

    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    if 't0' in optimizer.param_groups[0]:
        for prm in model.parameters():
            prm.data = temp[prm].clone()
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def main_exp(save_path, exp_name, model, optimizer, clip, train_loader, val_loader, test_loader, lang, scheduler=None):
    n_epochs = 100
    patience = 3

    d = datetime.now()
    strftime = d.strftime("%Y-%m-%d_%H-%M")

    writer = SummaryWriter(log_dir=f"{save_path}runs/{exp_name}/{strftime}/")
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train = []
    val_losses = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    runpath = save_path + 'runs/' + exp_name + '/' + strftime + '/'
    os.makedirs(runpath, exist_ok=True)
    f = open(runpath + 'results.txt', "w")
    file_path = runpath + exp_name + '.pt'
    pbar = tqdm(range(1, n_epochs), file=f)

    for epoch in pbar:
        train_ppl, train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip, scheduler)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(train_loss).mean())
            val_ppl, val_loss = eval_loop(val_loader, criterion_eval, model, optimizer)
            val_losses.append(np.asarray(val_loss).mean())

            log_values(writer, epoch, train_loss, train_ppl, "Train")
            log_values(writer, epoch, val_loss, val_ppl, "Validation")

            pbar.set_description("PPL: %f" % val_ppl)
            torch.save(model.state_dict(), file_path)

            if val_ppl < best_ppl:  # the lower, the better
                best_ppl = val_ppl
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model, optimizer)
    print(f'Test ppl: {final_ppl:2f}')
    f.write(f'Test ppl: {final_ppl:2f}')
    torch.save(best_model.state_dict(), file_path)
    f.close()
    writer.close()
