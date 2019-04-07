import os
import pandas as pd
import pickle

from .dl_data import download_unzip_data
from .load_data import make_train_test_data
from .train_cut_off import cutoff_like_test


DEFAULT_DIR = 'data'
DATA_POOL_DIR = os.path.join(DEFAULT_DIR, 'data_pool')
SUBMIT_DIR = os.path.join(DEFAULT_DIR, 'submission')

if not os.path.exists(DEFAULT_DIR):
    os.mkdir(DEFAULT_DIR)

if not os.path.exists(DATA_POOL_DIR):
    os.mkdir(DATA_POOL_DIR)

if not os.path.exists(SUBMIT_DIR):
    os.mkdir(SUBMIT_DIR)


class Dataset:
    '''
    baseデータの読み込み
    特徴量作成の親クラス
    '''

    def __init__(self):
        self._root_dir = DEFAULT_DIR
        self._data_dir = DATA_POOL_DIR
        self._submit_dir = SUBMIT_DIR

    def load_data(self, reproduce=False):
        self.df_p = 'base_df'

        if reproduce:
            return self._data_generate()
        else:
            if self.df_p in os.listdir(DATA_POOL_DIR):
                return self.load_pickel(self.df_p)

            else:
                return self._data_generate()

    def _data_generate(self):
        # データのDL/解凍
        download_unzip_data(self._root_dir)

        # 目的変数としてdead_duration・train_testをまとめたdfを作成
        train, test = make_train_test_data(self._root_dir)

        # testデータみたいに、不完全なフライトデータにする
        cut_train = cutoff_like_test(train, test)
        merged_df = pd.concat([cut_train, test], axis=0)

        self.write_pickel(merged_df, self.df_p)
        return merged_df

    def load_pickel(self, fname):
        fpath = os.path.join(self._data_dir, fname)
        with open(fpath, mode='rb') as f:
            rt = pickle.load(f)
        return rt

    def write_pickel(self, target, fname):
        fpath = os.path.join(self._data_dir, fname)
        with open(fpath, 'wb') as f:
            pickle.dump(target, f)

    def run(self, df, reproduce=False):
        # 再作成したいとき
        if reproduce:
            produced = self._create_feature(df)
            self.write_pickel(produced, self.pickel_name)
        else:
            # pickelがあれば再利用
            if os.path.exists(os.path.join(self._data_dir, self.pickel_name)):
                produced = self.load_pickel(self.pickel_name)
            else:
                # 初回は実行
                produced = self._create_feature(df)
                self.write_pickel(produced, self.pickel_name)

        return produced
