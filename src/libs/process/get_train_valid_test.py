import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def get_train_valid_test(all_df):
    train = all_df[(all_df['is_train'] == 1) & (all_df['is_valid'] == 0)]
    valid = all_df[all_df['is_valid'] == 1]
    test = all_df[all_df['is_train'] == 0]

    def drop_some_cols(_df):
        return _df.drop(['is_train', 'is_valid'], axis=1)
    train = drop_some_cols(train)
    valid = drop_some_cols(valid)
    test = drop_some_cols(test)

    return train, valid, test
