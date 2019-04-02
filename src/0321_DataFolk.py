# %%
import pandas as pd
import numpy as np
from processing.load_data import load_data

# %%
df = load_data('D://tmp/isid')

# %%
df.head()

# %%
train = df[df['is_train'] == 1]
test = df[df['is_train'] == 0]
df.shape, train.shape, test.shape

# %% engine_deadの分布
dead = df[df['engine_dead'] == 1]
dead['duration'].hist()
dead['duration'].describe()
# だいたい平均205時間くらいで死んでる、右肩なだらかな分布

# %% testのdurationの最後
test_last_dur = test.groupby('engine_no')['duration'].max()
test_last_dur.hist()
test_last_dur.describe()

# %% trainのengine_deadとtestのdurationラストの重ね合わせ
hist_df = pd.DataFrame({
    'train_dead': dead['duration'],
    'test_last_duration': test_last_dur
})
hist_df.plot.hist(alpha=0.5)
#

# %% カラムを見てみる

# %%
