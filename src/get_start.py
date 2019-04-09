import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from libs.Dataset import Dataset
from libs.EngineSumBase import EngineSumBase
from libs.EngineSumTimeGrad import EngineSumTimeGrad
from libs.EngineSumLastDur import EngineSumLastDur
from libs.EngineSumBasics import EngineSumBasics
from libs.standarzation import standarzation_x, encode_y, decode_z
from libs.premodel import split_learn_valid_test, standazation, decode_predict, mae_of_predict
from libs.submit import submitform

REGANARATE = True
TRAIN_CUTOFF = True
NUM_RESAMPLE_TRAIN = 5
SCALING = True
SUBMIT = True

# データ読み込み
df = Dataset().load_data(REGANARATE, TRAIN_CUTOFF, NUM_RESAMPLE_TRAIN)

# エンジン別特徴量の作成
summarized_df = EngineSumBase().create_feature(df, REGANARATE)
summarized_df = EngineSumTimeGrad().create_feature(
    df, summarized_df, REGANARATE)
summarized_df = EngineSumLastDur().create_feature(
    df, summarized_df, REGANARATE)
summarized_df = EngineSumBasics().create_feature(
    df, summarized_df, REGANARATE)

# train, valid, testに分割
train, valid, test = split_learn_valid_test(summarized_df)
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
# model = LinearRegression()
# model = Lasso()
# model = Ridge()
# model = LGBMRegressor()
# model = RandomForestRegressor()
model = SVR(gamma='auto')
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
