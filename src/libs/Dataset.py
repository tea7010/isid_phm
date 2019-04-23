import os
import pandas as pd
import pickle

from .io.dl_data import download_unzip_data
from .io.merge_train_test import merge_train_test
from .process.label_valid import label_valid, valid_engine_random
from .process.train_cut_off import cutoff_like_test


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

    methods:
        load_raw_data:
            生データを結合したDFを読み込む。過去に生成していればpickelで読み込み高速化する
        load/write_pickel:
            fnameのpickelを読み込む/書き込む

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
            else:
                # データのDL/解凍
                download_unzip_data(self._root_dir)

                # 目的変数としてdead_duration・train_testをまとめたdfを作成
                _df = merge_train_test(self._root_dir)

                self.write_pickel(_df, self.df_raw)
        return _df

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
