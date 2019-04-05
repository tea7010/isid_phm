import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from libs.Dataset import Dataset
from libs.EngineSumBase import EngineSumBase
from libs.EngineSumTimeGrad import EngineSumTimeGrad
from libs.EngineSumLastDur import EngineSumLastDur
from libs.EngineSumBasics import EngineSumBasics
from libs.premodel import standazation, decode_predict, mae_of_predict
from libs.submit import submitform

REGANARATE = True
FEATURE_REGANARATE = True
TEST_SIZE = 0.2
RANDOM_STRATE_VALID = 0

# データ読み込み
df = Dataset().load_data(REGANARATE)

# エンジン別特徴量の作成
summarized_df = EngineSumBase().create_feature(df, FEATURE_REGANARATE)
summarized_df = EngineSumTimeGrad().create_feature(
    df, summarized_df, FEATURE_REGANARATE)
summarized_df = EngineSumLastDur().create_feature(
    df, summarized_df, FEATURE_REGANARATE)
# summarized_df = EngineSumBasics().create_feature(df, summarized_df, FEATURE_REGANARATE)

# 正規化を行う
train_df = summarized_df[summarized_df['is_train']
                         == 1].drop(['is_train'], axis=1)
test_df = summarized_df[summarized_df['is_train']
                        == 0].drop(['is_train'], axis=1)

ss, colnames, train_df, test_df = standazation(train_df, test_df)


# モデルの学習用と評価用にtrainを分割
x_learn, x_valid, y_learn, y_valid = train_test_split(train_df.drop(['engine_dead'], axis=1),
                                                      train_df['engine_dead'], test_size=TEST_SIZE,
                                                      random_state=RANDOM_STRATE_VALID)

# モデル学習
model = SVR(gamma='auto')
# model = LGBMRegressor()
model.fit(x_learn, y_learn)

# モデル評価
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
valid_score = mae_of_predict(pre, x_valid, y_valid, ss, colnames)
print("Valid score: ", valid_score)

# testを予測
# pre = pd.DataFrame(model.predict(test_df), index=test_df.index)
# pre_inv = decode_predict(pre, test_df, ss, colnames)
# predit = submitform(test, pre_inv, True, output_path='data/submission')
