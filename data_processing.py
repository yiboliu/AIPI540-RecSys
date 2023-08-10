import pandas as pd
from sklearn.preprocessing import LabelEncoder


def convert_string_to_int(df, col1, col2):
    """Converts strings to ints by using label encoder provided by sklearn"""
    encoder = LabelEncoder()
    df[col2] = encoder.fit_transform(df[col1])
    return encoder, df


def read_csv(filename):
    """Read the csv file provided. Split the data rows by if it represents the user purchased/played that game.
    Only playing hours will make sense. This can also dedup, as most of the games are both played and purchased by the
    users. The label encoders are also returned for later use."""
    df = pd.read_csv(filename, header=None, names=['userId', 'gameTitle', 'purchased', 'hours', 'x'])
    purchased = df.loc[df['purchased'] == 'purchase']  # find the rows that are purchased, then we won't consider hours
    purchased = purchased[['userId', 'gameTitle']]
    data = df.loc[df['purchased'] == 'play']
    data = data[['userId', 'gameTitle', 'hours']]
    userid_encoder, data = convert_string_to_int(data, 'userId', 'userIdInt')
    gametitle_encoder, data = convert_string_to_int(data, 'gameTitle', 'gameTitleInt')
    return purchased, data, userid_encoder, gametitle_encoder


def get_features(data):
    """Calculates features of games, popularity and average play time, to be used for content filtering"""
    game_features = pd.DataFrame({'gameTitle': data['gameTitleInt'].unique()})
    game_features['popularity'] = game_features.apply(
        lambda x: len(
            data.loc[
                (data['gameTitleInt'] == x['gameTitle'])
            ]
        ), axis=1
    )

    def get_avg_play_time_game(gameTitle):
        specified = data.loc[(data['gameTitleInt'] == gameTitle)]
        sum = specified['hours'].sum()
        return sum / specified.shape[0]

    game_features['avg_play_time'] = game_features.apply(
        lambda x: get_avg_play_time_game(x['gameTitle']), axis=1
    )

    return game_features
