from sklearn.preprocessing import StandardScaler


ss = StandardScaler()
ss.fit(train_df)

colnames = train_df.columns
train_df = pd.DataFrame(ss.transform(train_df))
train_df.columns = colnames

test_df = pd.DataFrame(ss.transform(test_df))
test_df.columns = colnames
test_df.fillna(0, inplace=True)
test_df.drop(['engine_dead'], axis=1, inplace=True)

# %%
TEST_SIZE = 0.2
x_learn, x_valid, y_learn, y_valid = train_test_split(train_df.drop(['engine_dead'], axis=1),
                                                      train_df['engine_dead'], test_size=TEST_SIZE)
x_learn.shape, x_valid.shape, y_learn.shape, y_valid.shape

# %%


def regg_rul(y_pred, x_valid, colnames):
    inv_valid = pd.DataFrame(ss.inverse_transform(
        pd.concat([y_pred, x_valid], axis=1)))
    inv_valid.columns = colnames
    return inv_valid['engine_dead']


def cal_mae(pre, x_valid, y_valid, colnames):
    inv_y_pred = regg_rul(pre, x_valid, colnames)
    inv_y_valid = regg_rul(y_valid, x_valid, colnames)

    return mean_absolute_error(inv_y_valid, inv_y_pred)
