import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append('./src')

    from EvaluateCV import EvaluateCV

    params = {
        'fold_k': 5,
        'regenarate': True,
        'train_cutoff': True,
        'num_resample_train': 1,
        'scaling': True,
        'use_model': 'MLP',
        'model_params': {'hidden_layer_sizes': 12,
                         'activation': 'tanh',
                         'random_state': 3},
        'submit': True
    }

    EvaluateCV(params).run()
