import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from EvaluateCV import EvaluateCV

    params = {
        'fold_k': 5,
        'regenarate': True,
        'train_cutoff': True,
        'num_resample_train': 5,
        'scaling': True,
        'use_model': 'SVR',
        'model_params': {'gamma': 'auto'},
        'submit': True
    }

    EvaluateCV(params).run()
