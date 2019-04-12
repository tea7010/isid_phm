import os
import pandas as pd
import pickle

from .dl_data import download_unzip_data
from .load_data import load_data
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

    def load_data(self, reproduce=False, cutoff=True, num_train_sampling=1, write_pickel=True):
        '''
        データの読み込みをする

        input:
            reproduce: bool
                高速な読み込みのためにpickelに保存してるが、pickelの更新等で再生成させるときTrue
            cutoff: bool
                trainをtestの打ち切り数の推定分布から、似たような感じで打ち切るかどうか
                （validは常に打ち切りする）
            num_train_sampling: int > 0
                trainの打ち切りを同一エンジンに対して複数回行うときの回数

        returns:
            pandas DataFrame
                元のcsvをtrain, testを全ファイル繋げたもの
                cutoffオプション次第で、is_valid, trainの行は減っている
                他にも下記カラムが追加
                    engine_no: エンジンNo
                    duration: そのエンジンの累計フライト数
                    dead_duration: 死んだフライト数
                    engine_dead: そのフライトで死亡か生存か
                    is_train: trainだと1
                    is_valid: validだと1
        '''
        self.df_p = 'base_df'

        if reproduce:
            return self._data_generate(cutoff, num_train_sampling, write_pickel)
        else:
            if self.df_p in os.listdir(self._data_dir):
                return self.load_pickel(self.df_p)

            else:
                return self._data_generate(cutoff, num_train_sampling, write_pickel)

    def _data_generate(self, cutoff, num_train_sampling, write_pickel):
        # データのDL/解凍
        download_unzip_data(self._root_dir)

        # 目的変数としてdead_duration・train_testをまとめたdfを作成
        all_df = load_data(self._root_dir)
        train = all_df[(all_df['is_train'] == 1) & (all_df['is_valid'] != 1)]
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
