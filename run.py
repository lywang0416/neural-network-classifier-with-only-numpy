from hyperparameter_search import *
from utils import *
from train import *
from test import test

def main():
    # different search mode
    args = parse_arguments()
    search_mode = args.search_mode
    if search_mode == 'learning_rate':
        learning_rate_search(args)
    elif search_mode == 'reg_strength':
        reg_strength_search(args)
    elif search_mode == 'hidden_size':
        hidden_size_search(args)
    elif search_mode == 'all':
        hyper_parameter_search(args)
    else:
        raise ValueError(f'Unknown search mode {search_mode}')

if __name__ == "__main__":
    main()