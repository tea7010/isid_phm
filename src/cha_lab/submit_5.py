import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from run_reggression_model import run_reggression_model

    params = {
        'regenarate': True,
        'train_cutoff': True,
        'num_resample_train': 5,
        'scaling': True,
        'use_model': 'SVR',
        'model_params': {'gamma': 'auto'},
        'submit': True
    }

    valid_score = run_reggression_model(params)
