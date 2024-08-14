# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py
import torch
# Import everything from functions.py file
from functions import *

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    to_run = {'RNN': {'model': LM_RNN, 'optimizer': optim.SGD, 'run': True},
              'LSTM': {'model': LM_LSTM, 'optimizer': optim.SGD, 'run': True},
              'LSTM_DROP': {'model': LM_LSTM, 'optimizer': optim.SGD, 'emb_dropout': 0.1, 'out_dropout': 0.1,
                            'run': True},
              'LSTM_DROP_ADAMW': {'model': LM_LSTM, 'optimizer': optim.AdamW, 'lr': 0.001, 'emb_dropout': 0.1,
                                  'out_dropout': 0.1, 'run': True}}

    run_experiments(to_run)
