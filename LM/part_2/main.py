# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py
import torch
# Import everything from functions.py file
from functions import *

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    to_run = {'LSTM_WEIGHT_TYING_SGD': {'run': False, 'emb_dropout': 0, 'out_dropout': 0},
              'LSTM_WEIGHT_TYING_DROPOUT_SGD': {'run': False, 'variational_dropout': True},
              'LSTM_WEIGHT_TYING_DROPOUT_AVSGD': {'run': False, 'avgSGD': True}, }

    run_experiments(to_run)
