# %%
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
import os
import glob
import pandas as pd
import numpy as np
import pandas_bokeh
import matplotlib.pyplot as plt

pandas_bokeh.output_notebook()


# %%
root_dir = 'D://tmp/isid'
train_dir = os.path.join(root_dir, 'Train Files/Train Files')
test_dir = os.path.join(root_dir, 'Test Files/Test Files')
train_csv_list = glob.glob(train_dir + '/*.csv')
test_csv_list = glob.glob(test_dir + '/*.csv')

# %%
file_path = np.random.choice(train_csv_list)
sample_df = pd.read_csv(file_path, encoding='Shift-JIS')
sample_df.dtypes

# %%
sample_df.isnull().any()

# %%
sample_df.describe()

# %%
sample_df['Flight Regime'].unique()

# %%
# 各ファイルのフライト数とレジームをまとめてみる
cols = ['is_train', 'num_flight']
for i in range(1, 7):
    cols += ['reseme_%s' % i]
fs_df = pd.DataFrame(index=train_csv_list + test_csv_list, columns=cols)
fs_df['is_train'] = 0
fs_df.loc[train_csv_list, 'is_train'] = 1

for file_i in train_csv_list + test_csv_list:
    csv_i = pd.read_csv(file_i, encoding='Shift-JIS')
    n_flight = len(csv_i)
    fs_df.loc[file_i, 'num_flight'] = n_flight
    for i in range(1, 7):
        fs_df.loc[file_i, 'reseme_%s' % i] = len(
            csv_i[csv_i['Flight Regime'] == i])/n_flight

# %%
# フライト数のtrainとtestのヒストグラム
train_flight = fs_df['num_flight'].copy()
train_flight[fs_df[fs_df['is_train'] == 0].index] = np.nan
test_flight = fs_df['num_flight'].copy()
test_flight[fs_df[fs_df['is_train'] == 1].index] = np.nan
pd.DataFrame({'train': train_flight, 'test': test_flight}
             ).plot_bokeh(kind='hist', alpha=0.4)

# %%
# 各ファイルをひとつのDFにして、各時刻の残存フライト数を求める
df = pd.DataFrame()
for file_i in train_csv_list + test_csv_list:
    df_i = pd.read_csv(file_i, encoding='Shift-JIS')

    # trainとtestの区別
    train_or_test = os.path.basename(file_i).split('_')[0]
    if train_or_test == 'Train':
        df_i['is_train'] = 1
    else:
        df_i['is_train'] = 0

    # durationの作成、event_occuredにラベル付け
    df_i['duration'] = pd.Series(range(len(df_i)))
    df_i['engine_dead'] = 0
    df_i.loc[len(df_i)-1, 'engine_dead'] = 1

    # エンジンNo
    df_i['engine_no'] = os.path.basename(file_i)[len(train_or_test + '_'):-4]

    df = pd.concat([df, df_i], axis=0)

df.reset_index(inplace=True)
df.loc[df[df['is_train'] == 0].index, 'engine_dead'] = 0

# %%
print(df['is_train'].value_counts())
print(df['engine_dead'].value_counts())
print(df[df['is_train'] == 1]['engine_no'].nunique())

# %%
# kaplan-meier使ってみる
train = df[df['is_train'] == 1]
y_train, y_valid = train_test_split(train, test_size=0.2)

# %%
kmf = KaplanMeierFitter()
kmf.fit(y_train['duration'])

# %%
kmf.plot()

# %%
kmf.median_

# %%
kmf.plot_cumulative_hazard()

# %%
kmf.predict(y_valid['duration'])

# %%


# %%
def drop_some_cols(df):
    _df = df.drop(['index', 'Unnamed: 25', 'engine_no', 'is_train'], axis=1)
    _df = _df[[
        'Altitude', 'Power Setting (TRA)', 'BPR Bypass Ratio --', 'duration', 'engine_dead']]
    return _df


train_df = df[df['is_train'] == 1]
test_df = df[df['is_train'] == 0]

learn_df, valid_df = train_test_split(train_df, test_size=0.15)

# %%
# cox hazard model
cph = CoxPHFitter()
cph.fit(drop_some_cols(learn_df),
        duration_col='duration', event_col='engine_dead')

# %%
cph.print_summary()

# %%
predict_df = drop_some_cols(valid_df).drop(['engine_dead'], axis=1)
cph.predict_survival_function(predict_df)

# %%
predict_df = drop_some_cols(valid_df).drop(['engine_dead'], axis=1)
unconditioned_sf = cph.predict_survival_function(predict_df)

# %%

# %%
sample_i = 37942
cum_h = cph.predict_cumulative_hazard(valid_df)
cum_h[sample_i].plot_bokeh()

# %%
engine_i = valid_df.loc[sample_i, 'engine_no']
df[df['engine_no'] == engine_i]['duration']

# %%
df.columns

# %%
df['Altitude'].plot_bokeh(kind='hist')

# %%
