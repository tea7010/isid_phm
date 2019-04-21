import os
import pandas as pd
import pickle

from .io.dl_data import download_unzip_data
from .io.merge_train_test import merge_train_test
from .io.label_valid import label_valid, valid_engine_random
from .io.train_cut_off import cutoff_like_test


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

    def load_raw_data(self, reproduce=True):
        self.df_raw = 'raw_df'

        if reproduce:
            # データのDL/解凍
            download_unzip_data(self._root_dir)

            # 目的変数としてdead_duration・train_testをまとめたdfを作成
            _df = merge_train_test(self._root_dir)
        else:
            if self.df_raw in os.listdir(self._data_dir):
                _df = self.load_pickel(self.df_raw)
        return _df

    def load_data(self, reproduce=True, cutoff=True, num_train_sampling=1, write_pickel=False):
        self.df_p = 'base_df'
        if reproduce:
            return self._data_generate(reproduce, cutoff, num_train_sampling, write_pickel)
        else:
            if self.df_p in os.listdir(self._data_dir):
                return self.load_pickel(self.df_p)

            else:
                return self._data_generate(reproduce, cutoff, num_train_sampling, write_pickel)

    def _data_generate(self, reproduce, cutoff, num_train_sampling, write_pickel):
        all_df = self.load_raw_data(reproduce).copy()

        # validエンジンの指定
        valid_engine = valid_engine_random(all_df, 30)
        all_df = label_valid(all_df, valid_engine)

        train = all_df[(all_df['is_train'] == 1) &
                       (all_df['is_valid'] != 1)]
        test = all_df[all_df['is_train'] == 0]
        valid = all_df[all_df['is_valid'] == 1]

        if cutoff:
            # testデータみたいに、不完全なフライトデータにする
            cut_train = cutoff_like_test(train, train, num_train_sampling)
            merged_df = pd.concat([cut_train, test], axis=0)
        else:
            merged_df = pd.concat([train, test], axis=0)

        # validはカットオフ1回のみ実施
        cut_valid = cutoff_like_test(valid, test, 1)
        merged_df = pd.concat([merged_df, cut_valid], axis=0)

        if write_pickel:
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
        '''
        これを読み込んだ子クラスが実行するメソッド
        '''
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
