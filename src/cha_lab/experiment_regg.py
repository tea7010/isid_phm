import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from libs.EvaluateModels import EvaluateModels

    NUM = 20
    experiment_df = pd.DataFrame(index=range(NUM))
    for i in range(NUM):
        params = {
            'approach': 'Engine_summary_reggresion',
            'regenarate': True,
            'train_cutoff': True,
            'num_resample_train': i+1,
            'scaling': True,
            'use_model': 'SVR',
            'model_params': {'gamma': 'auto'},
        }

        Eval_i = EvaluateModels(params)
        Eval_i.run_hold_out()
        valid_score = Eval_i.holdout_score

        experiment_df.loc[i, 'Experiment_No'] = i
        experiment_df.loc[i, 'score'] = valid_score
        experiment_df.to_csv('result.csv')
