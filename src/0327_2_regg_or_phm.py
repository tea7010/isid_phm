# %%
from libs.processing import make_summarize_table
import pandas as pd
import numpy as np
from libs.load_data import make_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# %%
path = 'D://tmp/isid'
df, train, test = make_train_test_data(path)


# %% 各エンジンのレジーム数、推移みたいなのを見てみる
eg_i = np.random.choice(train['engine_no'].unique())
eg_df = train[train['engine_no'] == eg_i]

print(eg_i)
print('shape:', eg_df.shape)
eg_df['Flight Regime'].value_counts()

# %%
for col_i in eg_df.columns:
    sns.relplot(x='duration', y=col_i, hue='Flight Regime', data=eg_df)

# %%
eg_df['Power Setting (TRA)'].value_counts()

# %%
plt.clf()
for col_i in eg_df.columns:
    sns.relplot(x='duration', y=col_i, hue='Power Setting (TRA)', data=eg_df)


# %% 特徴量候補、各パラメータの直近50までの勾配、最初との差分
target_cols = ['NRc Corrected core speed rpm',
               'T30 Total temperature at HPC outlet ｰR',
               'T50 Total temperature at LPT outlet ｰR',
               'Nc Physical core speed rpm',
               'Ps30 Static pressure at HPC outlet psia',
               'BPR Bypass Ratio --',
               'htBleed (Bleed Enthalpy)']

summarize_df = pd.DataFrame(index=train['engine_no'].unique())

# %%
engine_list = train['engine_no'].unique()
for eg_i in engine_list:
    eg_df = train[train['engine_no'] == eg_i]

    for regime_i in [1, 2, 3, 4, 5, 6]:
        regime_df = eg_df[eg_df['Flight Regime'] == regime_i]

        if len(regime_df) == 0:
            continue

        for col_i in target_cols:
            summarize_df.loc[eg_i,
                             'engine_dead'] = regime_df['dead_duration'].iloc[-1]

            exac_process = 'Grad_last_50'
            colname = 'Regime_%s_%s_%s' % (regime_i, col_i, exac_process)
            summarize_df.loc[eg_i, colname] = cal_nearest_grad(
                regime_df, col_i)

            exac_process = 'Grad_start_last'
            colname = 'Regime_%s_%s_%s' % (regime_i, col_i, exac_process)
            summarize_df.loc[eg_i, colname] = cal_first_last_round_grad(
                regime_df, col_i)


# %%
eg_i = np.random.choice(train['engine_no'].unique())
eg_df = train[train['engine_no'] == eg_i]

regime_i = 4
regime_df = eg_df[eg_df['Flight Regime'] == regime_i]

col_i = target_cols[2]
regime_df[col_i]
col_i

# %%


def cal_nearest_grad(regime_df, col_i):
    REVIEW_TERM = 50
    x = regime_df['duration']
    y = regime_df[col_i]

    from_idx = (x - (x.iloc[-1] - REVIEW_TERM)).idxmin()
    to_idx = x.index[-1]
    grad = (y.loc[to_idx] - y.loc[from_idx])/(x.loc[to_idx] - x.loc[from_idx])
    return grad


def cal_first_last_round_grad(regime_df, col_i):
    ROUND_NUM = 3
    x = regime_df['duration']
    y = regime_df[col_i]

    x_from = x.iloc[:ROUND_NUM].mean()
    x_to = x.iloc[-ROUND_NUM:].mean()
    y_from = y.iloc[:ROUND_NUM].mean()
    y_to = y.iloc[-ROUND_NUM:].mean()
    grad = (y_to - y_from)/(x_to - x_from)
    return grad


# %%
eg_i = np.random.choice(train['engine_no'].unique())
eg_df = train[train['engine_no'] == eg_i]

print(eg_i)
print('shape:', eg_df.shape)
eg_df['Flight Regime'].value_counts()

for col_i in eg_df.columns:
    sns.relplot(x='duration', y=col_i, hue='Flight Regime', data=eg_df)

# %%
summarize_df.loc[eg_i]

# %%

# %%
summarize_df


# %%

# %%
test_summarize = make_summarize_table(test)
train_summarize = make_summarize_table(train)

test_summarize

# %%
train_summarize.isnull().any().sum()

# %%
test_summarize.isnull().any()

# %%
