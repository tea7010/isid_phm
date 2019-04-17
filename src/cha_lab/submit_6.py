from sklearn.metrics import mean_absolute_error
import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append('./src')

    from libs.Dataset import Dataset

    df = Dataset().load_data(reproduce=True, cutoff=False)

    # train, valid, testのデータに分割
    train = df[(df['is_train'] == 1) & (df['is_valid'] == 0)]
    valid = df[df['is_valid'] == 1]
    test = df[df['is_train'] == 0]

    dead_flight = train.groupby('engine_no')['dead_duration'].first()
    dead_mean = dead_flight.mean()

    # 各エンジンに対する死亡フライト数の予測値
    predict_dead_valid = pd.Series([dead_mean]*(valid['engine_no'].nunique()))

    # 本当のvalidの死亡フライト数
    true_dead_valid = valid.groupby(['engine_no'])['dead_duration'].first()
    mae = mean_absolute_error(true_dead_valid, predict_dead_valid)
    print('Valid score: ', mae)
