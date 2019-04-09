import pandas as pd
import numpy as np
import os
import pickle
import glob

RANDOM_SEED = 0
VALID_NUM_ENGINE = 30


def load_data(dir_path):
    # csvのpathリストの作成
    train_dir = os.path.join(dir_path, 'Train Files')
    test_dir = os.path.join(dir_path, 'Test Files')
    train_csv_list = glob.glob(train_dir + '/*.csv')
    test_csv_list = glob.glob(test_dir + '/*.csv')

    df = pd.DataFrame()
    for file_i in train_csv_list + test_csv_list:
        df_i = pd.read_csv(file_i, encoding='Shift-JIS')

        # trainとtestの区別
        train_or_test = os.path.basename(file_i).split('_')[0]
        if train_or_test == 'Train':
            df_i['is_train'] = 1
        else:
            df_i['is_train'] = 0

        # durationの作成、event_occuredにラベル付け
        df_i['duration'] = pd.Series(range(len(df_i)))
        df_i['engine_dead'] = 0
        df_i.loc[len(df_i)-1, 'engine_dead'] = 1

        # そのエンジンの死亡する期間
        df_i['dead_duration'] = df_i[df_i['engine_dead'] == 1]['duration'].max()

        # エンジンNo
        df_i['engine_no'] = os.path.basename(
            file_i)[:-4]

        df = pd.concat([df, df_i], axis=0)

    df.reset_index(inplace=True)

    # testのengine_deadは0
    df.loc[df[df['is_train'] == 0].index, 'engine_dead'] = 0

    # valid用のエンジンをランダムに選択
    df['is_valid'] = 0
    train_eg = df[df['is_train'] == 1]['engine_no'].unique()

    np.random.seed(RANDOM_SEED)
    valid_eg = np.random.choice(train_eg, VALID_NUM_ENGINE, replace=False)
    df.loc[df[df['engine_no'].isin(valid_eg)].index, 'is_valid'] = 1

    # いらないカラムの削除
    df.drop(['index', 'Unnamed: 25'], axis=1, inplace=True)
    return df
