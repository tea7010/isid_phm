# %%
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from processing.load_data import load_data
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# %%
df = load_data("D://tmp/isid")
train = df[df['is_train'] == 1]
test = df[df['is_train'] == 0]

# trainのengine_noをもとに学習と評価用に分ける
VALID_NUM_ENGINE = 50
train_eg = train['engine_no'].unique()
np.random.seed(0)
valid_eg = np.random.choice(train_eg, VALID_NUM_ENGINE)
learn_eg = np.setdiff1d(train_eg, valid_eg)
train_eg.shape, valid_eg.shape, learn_eg.shape

# %%
learn = train[train['engine_no'].isin(learn_eg)]
valid = train[train['engine_no'].isin(valid_eg)]
learn['engine_no'].nunique()

# 死んでないサンプル数を減らしてみる
alive = learn[learn['engine_dead'] == 0]
dead = learn[learn['engine_dead'] == 1]

DEG_NUM = len(dead)
np.random.seed(0)
deg_index = pd.Series(np.random.choice(alive.index, DEG_NUM))
deg_index = pd.concat([pd.Series(dead.index), deg_index])
learn = learn.loc[deg_index]

# %%
kmf = KaplanMeierFitter()
kmf.fit(durations=learn['duration'], event_observed=learn['engine_dead'])
kmf.plot()

# %%
valid_eg = valid['engine_no'].unique()
valid_summary = pd.DataFrame(index=valid_eg)
valid_summary['true_dead_duration'] = valid.groupby(
    'engine_no')['dead_duration'].first()
valid_summary

# %% とりあえずengine_deadの平均をモデルとしてみる
dead_dr = train.groupby('engine_no')['dead_duration'].mean().mean()
dead_dr

# %%
valid_summary['predict_dead_duration'] = dead_dr
mean_absolute_error(
    valid_summary['true_dead_duration'], valid_summary['predict_dead_duration'])

# %%
test_last_flight = test.groupby('engine_no')['dead_duration'].first()
rul_predict_test = dead_dr - test_last_flight
rul_predict_test[rul_predict_test < 0] = 0
rul_predict_test = pd.DataFrame(
    {'Predicted RUL': rul_predict_test})
rul_predict_test.to_csv('D://tmp/isid/C0002_25_3_2019.csv', index=False)

# %%

# %%
