import pandas as pd
from sklearn.preprocessing import StandardScaler


def standarzation_x(x_learn, x_valid, x_test):
    ss = StandardScaler()
    x_merged = pd.concat([x_learn, x_valid, x_test], axis=0)
    ss.fit(x_merged)

    xz_learn = pd.DataFrame(ss.transform(x_learn), columns=x_learn.columns)
    xz_valid = pd.DataFrame(ss.transform(x_valid), columns=x_valid.columns)
    xz_test = pd.DataFrame(ss.transform(x_test), columns=x_test.columns)

    return xz_learn, xz_valid, xz_test


def encode_y(y, m, std):
    z = (y - m)/std
    return z


def decode_z(z, m, std):
    y = z*std + m
    return y
