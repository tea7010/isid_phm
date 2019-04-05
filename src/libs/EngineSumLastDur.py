import os
import pandas as pd
import numpy as np
from .Dataset import Dataset


class EngineSumLastDur(Dataset):
    def __init__(self):
        super().__init__()

        _name = 'EngineSummary'
        _feature = 'last_dur'
        self.pickel_name = '%s_%s' % (_name, _feature)

    def create_feature(self, df, base, reproduce=False):
        add = self.run(df, reproduce)
        return pd.concat([base, add], axis=1)

    def _create_feature(self, df):
        # 各エンジンに対してsummarize
        engine_list = df['engine_no'].unique()
        summarize_df = pd.DataFrame(index=engine_list)

        for eg_i in engine_list:
            eg_df = df[df['engine_no'] == eg_i]

            # 最後のフライト数
            summarize_df.loc[eg_i,
                             'last_duration'] = eg_df['duration'].iloc[-1]

        return summarize_df
