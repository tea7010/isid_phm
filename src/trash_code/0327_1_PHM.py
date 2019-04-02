# %%
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from sklearn.decomposition import PCA
import os
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from libs.load_data import make_train_test_data
from libs.make_valid_data import make_valid_data
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# %%
root_path = 'D://tmp/isid'
df, train, test = make_train_test_data(root_path)
learn, valid = make_valid_data(train)
df.shape, test.shape, train.shape, learn.shape, valid.shape

# %% 多重線形性があるのでとりあえずdatafalkしてみる
df.columns

# %%
df.corr()

# %% PCAで次元圧縮してみる
N_COMPONENTS = 4
except_cols = ['is_train', 'duration', 'engine_dead',
               'dead_duration', 'engine_no']
df_pca_bef = df.drop(except_cols, axis=1)

pca = PCA(n_components=N_COMPONENTS)
pca.fit(df_pca_bef)

print(pca.explained_variance_ratio_.sum())
print(pca.explained_variance_ratio_)

df_pca_aft = pd.DataFrame(pca.transform(df_pca_bef))
df_pca_aft.columns = ['PCA_%s' % i for i in range(N_COMPONENTS)]

# %%
df_processed = df_pca_aft.copy()
df_processed[except_cols] = df[except_cols]
df_processed

# %% 公式の例データでPHMの練習
rossi = load_rossi()
rossi.head()

# %%

rossi_dataset = load_rossi()
cph = CoxPHFitter()
cph.fit(rossi_dataset, duration_col='week',
        event_col='arrest', show_progress=True)
cph.print_summary()

# %%
x = rossi_dataset.drop(['week', 'arrest'], axis=1)
cph.predict_survival_function(x, times=[5, 25])

# %%
cph.predict_cumulative_hazard(x)

# %% cox hazardちょい試す
train = df_processed[df_processed['is_train'] == 1]
test = df_processed[df_processed['is_train'] == 0]

learn, valid = make_valid_data(train)

alive = learn[learn['engine_dead'] == 0]
dead = learn[learn['engine_dead'] == 1]

DEG_NUM = len(dead)*2
np.random.seed(0)
deg_index = pd.Series(np.random.choice(alive.index, DEG_NUM))
deg_index = pd.concat([pd.Series(dead.index), deg_index])
learn = learn.loc[deg_index]

drop_cols = ['is_train', 'dead_duration', 'engine_no']
learn.drop(drop_cols, axis=1, inplace=True)
valid.drop(drop_cols, axis=1, inplace=True)

# %%
cph = CoxPHFitter()
cph.fit(learn, duration_col='duration', event_col='engine_dead')
cph.print_summary()

# %%
cph.predict_survival_function(valid)
# これだと各時点での生存確率がでるみたいなので、やはり生存分析にあてはめるにしても個体間の独立性（サマライズ）が必要

# %%
