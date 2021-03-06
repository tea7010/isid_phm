import os
import pandas as pd
from datetime import datetime


def submitform(test, predict, output=False, output_path=''):
    '''
    predict: そのエンジンの予測寿命値
        * エンジンNoの若い順に並んでいること
    '''
    test_last_flight = pd.DataFrame(test.groupby('engine_no')[
                                    'dead_duration'].first())
    predict.columns = ['dead_duration']
    predict.index = test_last_flight.index
    # 予測寿命から、最後のフライト数を引くと、余命
    rul_predict_test = predict - test_last_flight

    # 負の予測は0に置き換える
    rul_predict_test[rul_predict_test < 0] = 0
    # カラム名
    rul_predict_test.columns = ['Predicted RUL']

    if output:
        date = datetime.now().strftime('%d_%m_%Y')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = os.path.join(output_path, 'C0002_%s.csv' % date)
        rul_predict_test.to_csv(output_path, index=False)

    return rul_predict_test
