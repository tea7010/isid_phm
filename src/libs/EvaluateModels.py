import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from libs.Dataset import Dataset
from libs.process.label_valid import label_valid, valid_engine_random
from libs.process.train_cut_off import cutoff_like_test
from libs.process.standarzation import decode_z
from libs.io.submit import submitform

from libs.engine_summary import engine_summarize_reggression
from libs.tutorial_model import tutorial_model

FOLD_NUM = 5
VALID_ENGINE_NUM = 30

PROC = mp.cpu_count()


class EvaluateModels:
    def __init__(self, params):
        '''
        モデルを評価するための枠組み

        methods:
            run_cv: CVによるモデルの評価を実行
            run_hol_out: ホールドアウトによるモデルの評価を実行
            make_submit: 学習したモデルで提出用ファイルを出力
        property:
            self.cv_df: CVの結果のdf
            self.cv_score: CVの結果のスコア
            self.holdout_score: ホールドアウトの結果のスコア
            self.params: 各種設定パラメータ
            self.model: モデル
            self.x_{train/valid/test}
        '''
        self.params = params
        self.APPROACH = params['approach']
        self.REGENARATE = params['regenarate']
        self.TRAIN_CUT = params['train_cutoff']
        self.NUM_RESAMPLE_TRAIN = params['num_resample_train']
        self.FEATURE = params['feature']
        self.SCALING = params['scaling']
        self.USE_MODEL = params['use_model']
        self.MODEL_PARAMS = params['model_params']

        self.raw_df = Dataset().load_raw_data(reproduce=False)

    def run_cv(self, n_fold=FOLD_NUM):
        self.cv_df = pd.DataFrame(index=range(n_fold))

        train_eg = self.raw_df[self.raw_df['is_train']
                               == 1]['engine_no'].unique()
        self._eg_split = np.array_split(train_eg, n_fold)

        pool = mp.Pool(PROC)
        callback = pool.map(self._cv_i, range(n_fold))

        self.cv_score = sum(callback)/n_fold
        print('CV score: %f' % self.cv_score)

    def _cv_i(self, i):
        score_i = self._learn_evaluate(self._eg_split[i])
        print("%s-th score: %f" % (i+1, score_i))
        return score_i

    def run_hold_out(self):
        valid_eg = valid_engine_random(self.raw_df, VALID_ENGINE_NUM)

        self.holdout_score = self._learn_evaluate(valid_eg)
        print('Hold-out score: %f' % self.holdout_score)

    def _learn_evaluate(self, engine_list):
        all_df = label_valid(self.raw_df, engine_list)

        # カットオフ実行
        df = self._cutoff_data(
            all_df, self.TRAIN_CUT, self.NUM_RESAMPLE_TRAIN)

        # アプローチ
        self.model, self.x_learn, self.y_learn, self.x_valid, self.y_valid, self.x_test = self._model_approach(
            df, self.APPROACH)

        # モデル評価
        predict = pd.DataFrame(self.model.predict(
            self.x_valid), index=self.x_valid.index)
        if self.SCALING:
            score_i = mean_absolute_error(
                decode_z(predict, self.y_learn.mean(), self.y_learn.std()), self.y_valid)
        else:
            score_i = mean_absolute_error(self.y_valid, predict)
        return score_i

    def _cutoff_data(self, all_df, cutoff, num_resample):
        train = all_df[(all_df['is_train'] == 1) &
                       (all_df['is_valid'] != 1)]
        test = all_df[all_df['is_train'] == 0]
        valid = all_df[all_df['is_valid'] == 1]

        if cutoff:
            # testデータみたいに、不完全なフライトデータにする
            cut_train = cutoff_like_test(train, train, num_resample)
            merged_df = pd.concat([cut_train, test], axis=0)
        else:
            merged_df = pd.concat([train, test], axis=0)

        # validはカットオフ1回のみ実施
        cut_valid = cutoff_like_test(valid, test, 1)
        df = pd.concat([merged_df, cut_valid], axis=0)
        return df

    def _model_approach(self, df, approach_type):
        if approach_type == 'Engine_summary_reggresion':
            model, x_learn, y_learn, x_valid, y_valid, x_test = engine_summarize_reggression(
                df, self.REGENARATE, self.SCALING, self.USE_MODEL, self.MODEL_PARAMS, self.FEATURE)
        elif approach_type == 'Tutorial':
            model, x_learn, y_learn, x_valid, y_valid, x_test = tutorial_model(
                df)

        return model, x_learn, y_learn, x_valid, y_valid, x_test

    def make_submit(self):
        predict = pd.DataFrame(self.model.predict(
            self.x_test), index=self.x_test.index)

        if self.SCALING:
            predict = decode_z(
                predict, self.y_learn.mean(), self.y_learn.std())

        test_df = self.raw_df[self.raw_df['is_train'] == 0].copy()
        self.submit = submitform(test_df, predict, True,
                                 output_path='data/submission')
