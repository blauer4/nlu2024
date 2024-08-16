# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    to_run = {'RNN': {'model': ModelIAS, 'optimizer': optim.SGD, 'run': True},
              'LSTM': {'model': ModelIAS, 'optimizer': optim.SGD, 'run': True},
              'LSTM_DROP': {'model': ModelIAS, 'optimizer': optim.SGD, 'emb_dropout': 0.1, 'out_dropout': 0.1,
                            'run': True},
              'LSTM_DROP_ADAMW': {'model': ModelIAS, 'optimizer': optim.AdamW, 'lr': 0.001, 'emb_dropout': 0.1,
                                  'out_dropout': 0.1, 'run': True}}

    train_init(to_run)
