# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    to_run = {'Dropout': {'lr': 0.0001, 'dropout': 0.1, 'run': True},
              'NoDropout': {'lr': 0.0001, 'dropout': 0, 'run': True},
              'Bidirectional': {'lr': 0.0001, 'bidirectional': True, 'run': True},
              'Bidirectional_dropout': {'lr': 0.0001, 'bidirectional': True, 'dropout': 0.1, 'run': True}, },

    run_experiments(to_run)
