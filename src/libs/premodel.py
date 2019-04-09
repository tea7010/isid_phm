import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def split_learn_valid_test(all_df):
    train = all_df[(all_df['is_train'] == 1) & (all_df['is_valid'] == 0)]
    valid = all_df[all_df['is_valid'] == 1]
    test = all_df[all_df['is_train'] == 0]

    def drop_some_cols(_df):
        return _df.drop(['is_train', 'is_valid'], axis=1)
    train = drop_some_cols(train)
    valid = drop_some_cols(valid)
    test = drop_some_cols(test)

    return train, valid, test


def standazation(train_df, test_df):
    '''
    正規化を行うクラス

    trainの平均・分散でtrain/testを正規化する
    予測のときに逆正規化をする必要があるので、使用したクラスはreturnする    
    '''
    ss = StandardScaler()
    ss.fit(train_df)

    colnames = train_df.columns
    train_df = pd.DataFrame(ss.transform(train_df))
    train_df.columns = colnames
    train_df.fillna(0, inplace=True)

    test_df = pd.DataFrame(ss.transform(test_df))
    test_df.columns = colnames
    test_df.fillna(0, inplace=True)
    test_df.drop(['dead_duration'], axis=1, inplace=True)
    return ss, colnames, train_df, test_df


def decode_predict(y_pred, x_valid, ss, colnames):
    '''
    寿命予測のために、正規化の逆(平均を足して分散を掛ける作業をskleanのクラスがやってくれる)を行う
    '''
    inv_valid = pd.DataFrame(ss.inverse_transform(
        pd.concat([y_pred, x_valid], axis=1)))
    inv_valid.columns = colnames
    return inv_valid['dead_duration']


def mae_of_predict(pre, x_valid, y_valid, ss, colnames):
    '''
    validデータの真の値と予測値の絶対平均誤差を計算する
    '''
    inv_y_pred = decode_predict(pre, x_valid, ss, colnames)
    inv_y_valid = decode_predict(y_valid, x_valid, ss, colnames)

    return mean_absolute_error(inv_y_valid, inv_y_pred)
