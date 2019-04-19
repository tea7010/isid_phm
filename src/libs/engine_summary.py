from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from libs.get_train_valid_test import get_train_valid_test
from libs.standarzation import standarzation_x, encode_y, decode_z

from libs.engine_summarize.EngineSumBase import EngineSumBase
from libs.engine_summarize.EngineSumTimeGrad import EngineSumTimeGrad
from libs.engine_summarize.EngineSumLastDur import EngineSumLastDur
from libs.engine_summarize.EngineSumBasics import EngineSumBasics


def engine_summarize_reggression(df, REGENARATE, SCALING, USE_MODEL, MODEL_PARAMS):
    # エンジン別特徴量の作成
    summarized_df = _make_feature(df, REGENARATE)

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

        model = _regg_model(x_learn, yz_learn, USE_MODEL, MODEL_PARAMS)
        return model, x_learn, y_learn, x_valid, y_valid, x_test
    else:
        model = _regg_model(x_learn, y_learn, USE_MODEL, MODEL_PARAMS)
        return model, x_learn, y_learn, x_valid, y_valid, x_test


def _make_feature(df, REGENARATE):
    summarized_df = EngineSumBase().create_feature(df, REGENARATE)
    summarized_df = EngineSumTimeGrad().create_feature(
        df, summarized_df, REGENARATE)
    summarized_df = EngineSumLastDur().create_feature(
        df, summarized_df, REGENARATE)
    summarized_df = EngineSumBasics().create_feature(
        df, summarized_df, REGENARATE)
    return summarized_df


def _regg_model(x_learn, yz_learn, USE_MODEL, MODEL_PARAMS):
    model_ex_dict = {
        'lm': 'LinearRegressio(**MODEL_PARAMS)',
        'Lasso': 'Lasso(**MODEL_PARAMS)',
        'Ridge': 'Ridge(**MODEL_PARAMS)',
        'LGB': 'LGBMRegressor(**MODEL_PARAMS)',
        'RF': 'RandomForestRegressor(**MODEL_PARAMS)',
        'SVR': 'SVR(**MODEL_PARAMS)',
        'MLP': 'MLPRegressor(**MODEL_PARAMS)'
    }

    model = eval(model_ex_dict[USE_MODEL])
    model.fit(x_learn, yz_learn)
    return model
