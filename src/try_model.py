# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from libs.dl_data import download_unzip_data
from libs.load_data import make_train_test_data, make_valid_data
from libs.train_cut_off import cutoff_like_test
from libs.processing import make_summarize_table
from libs.premodel import standazation, decode_predict, cal_mae
from libs.submit import submitform

# %%
DEFAULT_DIR = 'data'

# データのDL/解凍
download_unzip_data(DEFAULT_DIR)

# 目的変数としてdead_duration・train_testをまとめたdfを作成
df, train, test = make_train_test_data(DEFAULT_DIR)

# testデータみたいに、不完全なフライトデータにする
train = cutoff_like_test(train, test)

# 各エンジンNoに対して集計行を作成
train_df = make_summarize_table(train)
test_df = make_summarize_table(test)

# 正規化を行う
ss, colnames, train_df, test_df = standazation(train_df, test_df)

# モデルの学習用と評価用にtrainを分割
TEST_SIZE = 0.2
x_learn, x_valid, y_learn, y_valid = train_test_split(train_df.drop(['engine_dead'], axis=1),
                                                      train_df['engine_dead'], test_size=TEST_SIZE,
                                                      random_state=0)

# モデル学習
model = SVR()
# model = LGBMRegressor()
model.fit(x_learn, y_learn)

# モデル評価
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
valid_score = cal_mae(pre, x_valid, y_valid, ss, colnames)
print("Valid score: ", valid_score)

# testを予測
pre = pd.DataFrame(model.predict(test_df), index=test_df.index)
pre_inv = decode_predict(pre, test_df, ss, colnames)
predit = submitform(test, pre_inv, True, output_path=DEFAULT_DIR)


# %%
