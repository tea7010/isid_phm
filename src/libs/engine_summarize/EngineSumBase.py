import os
import pandas as pd
import numpy as np
from ..Dataset import Dataset


class EngineSumBase(Dataset):
    def __init__(self):
        super().__init__()

        _name = 'EngineSummary'
        _feature = 'base'
        self.pickel_name = '%s_%s' % (_name, _feature)

    def create_feature(self, df, reproduce=False):
        return self.run(df, reproduce)

    def _create_feature(self, df):
        # 各エンジンに対してsummarize
        engine_list = df['engine_no'].unique()
        summarize_df = pd.DataFrame(index=engine_list)

        for eg_i in engine_list:
            eg_df = df[df['engine_no'] == eg_i]

            # 目的変数
            summarize_df.loc[eg_i,
                             'dead_duration'] = eg_df['dead_duration'].iloc[-1]

            # train or test
            summarize_df.loc[eg_i,
                             'is_train'] = eg_df['is_train'].iloc[-1]

            # train or test
            summarize_df.loc[eg_i,
                             'is_valid'] = eg_df['is_valid'].iloc[-1]
        return summarize_df
