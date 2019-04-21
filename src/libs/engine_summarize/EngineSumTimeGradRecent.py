import pandas as pd
import numpy as np
from ..Dataset import Dataset


class EngineSumTimeGradRecent(Dataset):
    def __init__(self):
        super().__init__()

        _name = 'EngineSummary'
        _feature = 'TimeGradient'
        self.pickel_name = '%s_%s' % (_name, _feature)

    def create_feature(self, df, base, reproduce=False):
        add = self.run(df, reproduce)
        return pd.concat([base, add], axis=1)

    def _create_feature(self, df):
        TARGET_COLS = ['NRc Corrected core speed rpm',
                       'T30 Total temperature at HPC outlet ｰR',
                       'T50 Total temperature at LPT outlet ｰR',
                       'Nc Physical core speed rpm',
                       'Ps30 Static pressure at HPC outlet psia',
                       'BPR Bypass Ratio --',
                       'htBleed (Bleed Enthalpy)']

        # 各エンジンに対してsummarize
        engine_list = df['engine_no'].unique()
        summarize_df = pd.DataFrame(index=engine_list)

        for eg_i in engine_list:
            eg_df = df[df['engine_no'] == eg_i]

            # 各レジームに対して集計
            for regime_i in [1, 2, 3, 4, 5, 6]:
                regime_df = eg_df[eg_df['Flight Regime'] == regime_i]

                if len(regime_df) < 11:
                    continue

                for col_i in TARGET_COLS:
                    # 直近の勾配
                    exac_process = 'Grad_last_50'
                    colname = 'Regime_%s_%s_%s' % (
                        regime_i, col_i, exac_process)
                    summarize_df.loc[eg_i, colname] = self._cal_nearest_grad(
                        regime_df, col_i)
        return summarize_df

    def _cal_nearest_grad(self, regime_df, col_i):
        # REVIEW_TERM = 50
        REVIEW_TERM = round(len(regime_df)/3)
        x = regime_df['duration']
        y = regime_df[col_i]

        from_idx = (x - (x.iloc[-1] - REVIEW_TERM)).idxmin()
        to_idx = x.index[-1]
        grad = (y.loc[to_idx] - y.loc[from_idx]) / \
            (x.loc[to_idx] - x.loc[from_idx])
        return grad

    def _cal_first_last_round_grad(self, regime_df, col_i):
        ROUND_NUM = 10
        x = regime_df['duration']
        y = regime_df[col_i]

        x_from = x.iloc[:ROUND_NUM].mean()
        x_to = x.iloc[-ROUND_NUM:].mean()
        y_from = y.iloc[:ROUND_NUM].mean()
        y_to = y.iloc[-ROUND_NUM:].mean()
        grad = (y_to - y_from)/(x_to - x_from)
        return grad
