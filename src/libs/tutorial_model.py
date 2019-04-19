import pandas as pd


class TutorialModel:
    def __init__(self):
        pass

    def fit(self, df):
        dead_flight = df.groupby('engine_no')['dead_duration'].first()
        self.dead_mean = dead_flight.mean()

    def predict(self, df):
        return [self.dead_mean]*(len(df))


def tutorial_model(df):
    train = df[(df['is_train'] == 1) &
               (df['is_valid'] != 1)]
    test = df[df['is_train'] == 0]
    valid = df[df['is_valid'] == 1]

    model = TutorialModel()
    model.fit(train)

    x_learn = 0
    y_learn = 0
    x_valid = valid.groupby(['engine_no'])['dead_duration'].first()
    y_valid = valid.groupby(['engine_no'])['dead_duration'].first()
    x_test = test.groupby(['engine_no'])['dead_duration'].first()

    return model, x_learn, y_learn, x_valid, y_valid, x_test
