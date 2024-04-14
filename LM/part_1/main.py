# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py
import torch
# Import everything from functions.py file
from functions import *

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    to_run = {0: {'name': 'RNN', 'run': True}, 1: {'name': 'LSTM', 'run': True},
              2: {'name': 'LSTM_DROP', 'run': True},
              3: {'name': 'LSTM_DROP_ADAMW', 'run': True}}

    run_experiments(to_run)
