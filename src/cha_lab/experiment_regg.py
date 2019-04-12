import pandas as pd
import sys
sys.path.append('D://isid_phm/src/')

if __name__ == "__main__":
    from run_reggression_model import run_reggression_model

    NUM = 20
    experiment_df = pd.DataFrame(index=range(NUM))
    for i in range(NUM):
        params = {
            'regenarate': True,
            'train_cutoff': True,
            'num_resample_train': i+1,
            'scaling': True,
            'use_model': 'SVR',
            'model_params': {'gamma': 'auto'},
            'submit': False
        }

        valid_score = run_reggression_model(params)

        experiment_df.loc[i, 'Experiment_No'] = i
        experiment_df.loc[i, 'score'] = valid_score
        experiment_df.to_csv('result.csv')
