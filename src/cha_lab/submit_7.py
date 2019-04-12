import pandas as pd
import sys
sys.path.append('D://isid_phm/src/')

if __name__ == "__main__":
    from run_reggression_model import run_reggression_model

    params = {
        'regenarate': True,
        'train_cutoff': True,
        'num_resample_train': 12,
        'scaling': True,
        'use_model': 'SVR',
        'model_params': {'gamma': 'auto'},
        'submit': True
    }

    valid_score = run_reggression_model(params)
