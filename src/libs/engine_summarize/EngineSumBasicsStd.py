import os
import pandas as pd
import numpy as np
from ..Dataset import Dataset


class EngineSumBasicsStd(Dataset):
    def __init__(self):
        super().__init__()

        _name = 'EngineSummary'
        _feature = 'describe'
        self.pickel_name = '%s_%s' % (_name, _feature)

    def create_feature(self, df, base, reproduce=False):
        add = self.run(df, reproduce)
        return pd.concat([base, add], axis=1)

    def _create_feature(self, df):
        TARGET_COLS = ['Altitude',
                       'Mach #',
                       'Power Setting (TRA)',
                       'T2 Total temperature at fan inlet ｰR',
                       'T24 Total temperature at LPC outlet ｰR',
                       'T30 Total temperature at HPC outlet ｰR',
                       'T50 Total temperature at LPT outlet ｰR',
                       'P2 Pressure at fan inlet psia',
                       'P15 Total pressure in bypass-duct psia',
                       'P30 Total pressure at HPC outlet psia',
                       'Nf Physical fan speed rpm',
                       'Nc Physical core speed rpm',
                       'epr Engine pressure ratio (P50/P2) --',
                       'Ps30 Static pressure at HPC outlet psia',
                       'phi Ratio of fuel flow to Ps30 pps/psi',
                       'NRf Corrected fan speed rpm',
                       'NRc Corrected core speed rpm',
                       'BPR Bypass Ratio --',
                       'farB Burner fuel-air ratio --',
                       'htBleed (Bleed Enthalpy)',
                       'Nf_dmd Demanded fan speed rpm',
                       'PCNfR_dmd Demanded corrected fan speed rpm',
                       'W31 HPT coolant bleed lbm/s',
                       'W32 LPT coolant bleed lbm/s']

        # 各エンジンに対してsummarize
        engine_list = df['engine_no'].unique()
        summarize_df = pd.DataFrame(index=engine_list)

        for eg_i in engine_list:
            eg_df = df[df['engine_no'] == eg_i]

            for regime_i in [1, 2, 3, 4, 5, 6]:
                regime_df = eg_df[eg_df['Flight Regime'] == regime_i]

                if len(regime_df) < 5:
                    continue

                for col_i in TARGET_COLS:
                    colname = 'Regime_%s_%s_%s' % (
                        regime_i, col_i, 'std')
                    summarize_df.loc[eg_i, colname] = regime_df[col_i].std()
        return summarize_df
