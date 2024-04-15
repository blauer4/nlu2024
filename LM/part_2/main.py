# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py
import torch
# Import everything from functions.py file
from functions import *

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    to_run = {4: {'name': 'LSTM_WEIGHT_TYING_SGD', 'run': True},
              5: {'name': 'LSTM_WEIGHT_TYING_DROPOUT_SGD', 'run': True},
              6: {'name': 'LSTM_WEIGHT_TYING_DROPOUT_AVSGD', 'run': True}}

    run_experiments(to_run)
