# %%
from sklearn.neighbors.kde import KernelDensity
from libs.submit import submitform
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from libs.load_data import make_train_test_data, make_valid_data
from libs.processing import make_summarize_table

# %%
df, train, test = make_train_test_data('D://tmp/isid')
df.shape, train.shape, test.shape

# %% trainを途中でちょん切る
# testのdead_durationからkdeしてサンプリング


def cutoff_like_test(df, test):
    kd = KernelDensity()
    test_dur = test.groupby('engine_no')['dead_duration'].first()
    kd.fit(np.array(test_dur).reshape(-1, 1))

    np.random.seed(0)
    cutoff_df = pd.DataFrame()
    for eg_i in df['engine_no'].unique():
        df_i = df[df['engine_no'] == eg_i]

        cutoff_pos = int(round(kd.sample(1)[0][0]))
        while len(df_i) < cutoff_pos:
            cutoff_pos = int(round(kd.sample(1)[0][0]))

        cutoff_df = pd.concat([cutoff_df, df_i.iloc[:cutoff_pos]], axis=0)
    return cutoff_df


# %%
learn, valid = make_valid_data(train)

# %%
train_df = make_summarize_table(learn)
valid_df = make_summarize_table(cutoff_like_test(valid, test))
test_df = make_summarize_table(test)

# %%
# 正規化
ss = StandardScaler()
ss.fit(train_df)

colnames = train_df.columns
train_df = pd.DataFrame(ss.transform(train_df))
train_df.columns = colnames
train_df.fillna(0, inplace=True)

valid_df = pd.DataFrame(ss.transform(valid_df))
valid_df.columns = colnames
valid_df.fillna(0, inplace=True)

test_df = pd.DataFrame(ss.transform(test_df))
test_df.columns = colnames
test_df.fillna(0, inplace=True)
test_df.drop(['engine_dead'], axis=1, inplace=True)

# %%
# x = pd.concat([train_df.drop(['engine_dead'], axis=1),
#                test_df.drop(['engine_dead'], axis=1)], axis=0)
# y = train_df['engine_dead']

# %%
# x.shape, y.shape, train_df.shape, test_df.shape

# %%

# %%
TEST_SIZE = 0.2
x_learn, x_valid, y_learn, y_valid = train_test_split(train_df.drop(['engine_dead'], axis=1),
                                                      train_df['engine_dead'], test_size=TEST_SIZE,
                                                      random_state=0)
x_learn.shape, x_valid.shape, y_learn.shape, y_valid.shape

# %%
# x_learn = train_df.drop(['engine_dead'], axis=1)
# y_learn = train_df['engine_dead']
# x_learn = valid_df.drop(['engine_dead'], axis=1)
# y_learn = valid_df['engine_dead']

# %%


def regg_rul(y_pred, x_valid, colnames):
    inv_valid = pd.DataFrame(ss.inverse_transform(
        pd.concat([y_pred, x_valid], axis=1)))
    inv_valid.columns = colnames
    return inv_valid['engine_dead']


def cal_mae(pre, x_valid, y_valid, colnames):
    inv_y_pred = regg_rul(pre, x_valid, colnames)
    inv_y_valid = regg_rul(y_valid, x_valid, colnames)

    return mean_absolute_error(inv_y_valid, inv_y_pred)


# %%
dead_dr = train.groupby('engine_no')['dead_duration'].mean().mean()
pre = pd.Series([dead_dr]*len(x_valid), index=x_valid.index)
mean_absolute_error(pre, regg_rul(y_valid, x_valid, colname))

# %% linear
model = LinearRegression()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colname)

# %%
model = SVR()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)


# %%
model = RandomForestRegressor()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)

# %%
model = LassoCV()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)

# %%
model = Ridge()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)

# %%
model = Lasso()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)

# %%
model = lgb.LGBMRegressor()
model.fit(x_learn, y_learn)
pre = pd.DataFrame(model.predict(x_valid), index=x_valid.index)
cal_mae(pre, x_valid, y_valid, colnames)


# %%
model = SVR()
model.fit(x_learn, y_learn)

pre = pd.DataFrame(model.predict(test_df), index=test_df.index)
pre_inv = regg_rul(pre, test_df, colnames)
predit = submitform(test, pre_inv, True, output_path='D://tmp/isid')

# %%
test_df

# %%

# %%
colnames

# %%
predit

# %%
x_learn.shape, x_valid.shape

# %%
