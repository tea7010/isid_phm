import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from libs.Dataset import Dataset
from libs.engine_summarize.EngineSumBase import EngineSumBase
from libs.engine_summarize.EngineSumTimeGrad import EngineSumTimeGrad
from libs.engine_summarize.EngineSumLastDur import EngineSumLastDur
from libs.engine_summarize.EngineSumBasics import EngineSumBasics
from libs.standarzation import standarzation_x, encode_y, decode_z
from libs.io.label_valid import label_valid, valid_engine_random
from libs.io.train_cut_off import cutoff_like_test
from libs.get_train_valid_test import get_train_valid_test
from libs.io.submit import submitform


class EvaluateCV:
    def __init__(self, params):
        self.params = params
        self.CV = params['cv']
        self.FOLD_NUM = params['fold_k']
        self.REGENARATE = params['regenarate']
        self.TRAIN_CUT = params['train_cutoff']
        self.NUM_RESAMPLE_TRAIN = params['num_resample_train']
        self.SCALING = params['scaling']
        self.USE_MODEL = params['use_model']
        self.MODEL_PARAMS = params['model_params']
        self.SUBMIT = params['submit']

        # 生データの読み込み
        self.raw_df = Dataset().load_raw_data()
        self.cv_df = pd.DataFrame(index=range(self.FOLD_NUM))

    def run(self):
        # trainのエンジンNo
        train_eg = self.raw_df[self.raw_df['is_train']
                               == 1]['engine_no'].unique()
        eg_split = np.array_split(train_eg, self.FOLD_NUM)

        if self.CV:
            for i in range(self.FOLD_NUM):
                # validの作成(is_valid列にラベリング)
                all_df = label_valid(self.raw_df, eg_split[i])

                # カットオフ実行
                df = self._cutoff_data(
                    all_df, self.TRAIN_CUT, self.NUM_RESAMPLE_TRAIN)

                # アプローチ
                model, x_learn, y_learn, x_valid, y_valid, x_test = self._engine_summarize_regg(
                    df, self.REGENARATE, self.SCALING, self.USE_MODEL)

                # モデル評価
                predict = pd.DataFrame(model.predict(
                    x_valid), index=x_valid.index)
                if self.SCALING:
                    score_i = mean_absolute_error(
                        decode_z(predict, y_learn.mean(), y_learn.std()), y_valid)
                else:
                    score_i = mean_absolute_error(y_valid, predict)

                print('%s/%s score: %s' % (i+1, self.FOLD_NUM, score_i))
                self.cv_df.loc[i, 'score'] = score_i

            self.cv_score = self.cv_df['score'].sum()/self.FOLD_NUM
            print('CV score: %f' % self.cv_score)

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

    def _engine_summarize_regg(self, df, REGENARATE, SCALING, USE_MODEL):
        # エンジン別特徴量の作成
        summarized_df = EngineSumBase().create_feature(df, REGENARATE)
        summarized_df = EngineSumTimeGrad().create_feature(
            df, summarized_df, REGENARATE)
        summarized_df = EngineSumLastDur().create_feature(
            df, summarized_df, REGENARATE)
        summarized_df = EngineSumBasics().create_feature(
            df, summarized_df, REGENARATE)

        # train, valid, testに分割
        train, valid, test = get_train_valid_test(summarized_df)
        x_learn = train.drop(['dead_duration'], axis=1).fillna(0)
        y_learn = train['dead_duration'].fillna(0)
        x_valid = valid.drop(['dead_duration'], axis=1).fillna(0)
        y_valid = valid['dead_duration'].fillna(0)
        x_test = test.drop(['dead_duration'], axis=1).fillna(0)

        # trainで正規化を行う
        if SCALING:
            x_learn, x_valid, x_test = standarzation_x(
                x_learn, x_valid, x_test)
            yz_learn = encode_y(y_learn, y_learn.mean(), y_learn.std())
            model = self._regg_model(x_learn, yz_learn, USE_MODEL)
            return model, x_learn, y_learn, x_valid, y_valid, x_test
        else:
            model = self._regg_model(x_learn, y_learn)
            return model, x_learn, y_learn, x_valid, y_valid, x_test

    def _regg_model(self, x_learn, yz_learn, USE_MODEL):
        model_ex_dict = {
            'lm': 'LinearRegressio(**self.MODEL_PARAMS)',
            'Lasso': 'Lasso(**self.MODEL_PARAMS)',
            'Ridge': 'Ridge(**mdel_params)',
            'LGB': 'LGBMRegressor(**self.MODEL_PARAMS)',
            'RF': 'RandomForestRegressor(**self.MODEL_PARAMS)',
            'SVR': 'SVR(**self.MODEL_PARAMS)',
            'MLP': 'MLPRegressor(**self.MODEL_PARAMS)'
        }

        model = eval(model_ex_dict[USE_MODEL])
        model.fit(x_learn, yz_learn)
        return model
