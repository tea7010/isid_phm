import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity


def cutoff_like_test(df, test, reuse_num=1):
    '''
    testと同じような感じでdfを後ろから、いい感じにちょん切る関数

    1. testのdead_duraitonを、カーネル密度推定
    2. その得られたtestの推定分布からランダムサンプリングして、dfをちょん切る

    reuse_num > 1だと、同じサンプルからもう一回カットオフサンプルを生成
    '''
    kd = KernelDensity()
    test_dur = test.groupby('engine_no')['dead_duration'].first()
    kd.fit(np.array(test_dur).reshape(-1, 1))

    cutoff_df = pd.DataFrame()
    for i in range(reuse_num):
        cutoff_df = _reuse_engine(df, kd, cutoff_df, i)

    return cutoff_df


def _reuse_engine(df, kd, cutoff_df, random_seed):
    np.random.seed(random_seed)
    for eg_i in df['engine_no'].unique():
        df_i = df[df['engine_no'] == eg_i].copy()
        df_i['engine_no'] = '%s_reuse_%s' % (eg_i, random_seed)

        cutoff_pos = int(round(kd.sample(1)[0][0]))
        while len(df_i) < cutoff_pos:
            cutoff_pos = int(round(kd.sample(1)[0][0]))

        cutoff_df = pd.concat([cutoff_df, df_i.iloc[:cutoff_pos]], axis=0)

    return cutoff_df
