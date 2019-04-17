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
from libs.get_train_valid_test import get_train_valid_test
from libs.io.submit import submitform


def run_reggression_model(params):
    REGENARATE = params['regenarate']
    TRAIN_CUT = params['train_cutoff']
    NUM_RESAMPLE_TRAIN = params['num_resample_train']
    SCALING = params['scaling']
    USE_MODEL = params['use_model']
    model_params = params['model_params']
    SUBMIT = params['submit']

    df = Dataset().load_data(REGENARATE, TRAIN_CUT, NUM_RESAMPLE_TRAIN)

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
        x_learn, x_valid, x_test = standarzation_x(x_learn, x_valid, x_test)
        yz_learn = encode_y(y_learn, y_learn.mean(), y_learn.std())

    # モデル学習
    model_ex_dict = {
        'lm': 'LinearRegressio(**model_params)',
        'Lasso': 'Lasso(**model_params)',
        'Ridge': 'Ridge(**model_params)',
        'LGB': 'LGBMRegressor(**model_params)',
        'RF': 'RandomForestRegressor(**model_params)',
        'SVR': 'SVR(**model_params)',
        'MLP': 'MLPRegressor(**model_params)'
    }

    model = eval(model_ex_dict[USE_MODEL])
    model.fit(x_learn, yz_learn)

    # モデル評価
    predict_z = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
    valid_score = mean_absolute_error(
        decode_z(predict_z, y_learn.mean(), y_learn.std()), y_valid)
    print('Valid score:', valid_score)

    # testを予測
    if SUBMIT:
        predict_z = pd.DataFrame(model.predict(x_test), index=x_test.index)
        predict_y = decode_z(predict_z, y_learn.mean(), y_learn.std())

        test_df = df[df['is_train'] == 0].copy()
        predit = submitform(test_df, predict_y, True,
                            output_path='data/submission')

    return valid_score
