import pandas as pd
import numpy as np
import os
import pickle
import glob

RANDOM_SEED = 0
VALID_NUM_ENGINE = 50


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
    df.loc[df[df['is_train'] == 0].index, 'engine_dead'] = 0

    # いらないカラムの削除
    df.drop(['index', 'Unnamed: 25'], axis=1, inplace=True)
    return df


def make_train_test_data(dir_path):
    df = load_data(dir_path)
    train = df[df['is_train'] == 1].copy()
    test = df[df['is_train'] == 0].copy()
    return train, test


def make_valid_data(train):
    train_eg = train['engine_no'].unique()
    np.random.seed(RANDOM_SEED)
    valid_eg = np.random.choice(train_eg, VALID_NUM_ENGINE)
    learn_eg = np.setdiff1d(train_eg, valid_eg)

    learn = train[train['engine_no'].isin(learn_eg)]
    valid = train[train['engine_no'].isin(valid_eg)]
    return learn, valid
