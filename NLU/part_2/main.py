# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    to_run = {'BertJoint': {'dropout': 0, 'run': True},
              'BertJoint_dropout': {'dropout': 0.1, 'run': False}, }

    run_experiments(to_run)
