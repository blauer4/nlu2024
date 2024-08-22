# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    to_run = {'Baseline': {'run': False, 'n_runs': 5},
              'Dropout': {'dropout': 0.1, 'run': False, 'n_runs': 5},
              'Bidirectional': {'bidirectional': True, 'run': False, 'n_runs': 5},
              'Bidirectional_dropout': {'bidirectional': True, 'dropout': 0.1, 'run': False, 'n_runs': 5}, }

    run_experiments(to_run)
