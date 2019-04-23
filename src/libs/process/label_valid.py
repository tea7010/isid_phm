import numpy as np

RANDOM_SEED = 0
VALID_NUM_ENGINE = 30


def label_valid(df, valid_engine):
    df['is_valid'] = 0
    df.loc[df[df['engine_no'].isin(valid_engine)].index, 'is_valid'] = 1
    return df


def valid_engine_random(df, num_engine):
    # valid用のエンジンをランダムに選択
    train_eg = df[df['is_train'] == 1]['engine_no'].unique()

    np.random.seed(RANDOM_SEED)
    valid_eg = np.random.choice(train_eg, num_engine, replace=False)
    return valid_eg
