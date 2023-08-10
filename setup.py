import pickle
import os

from data_processing import read_csv, get_features

if __name__ == "__main__":
    purchased, data, userid_encoder, gametitle_encoder = read_csv('steam-200k.csv')
    game_features = get_features(data)
    if not os.path.exists('temp'):
        os.mkdir('temp')
    purchased.to_csv('temp/purchased.csv')
    data.to_csv('temp/data.csv')
    game_features.to_csv('temp/game_features.csv')
    with open('temp/userid_encoder.pkl', 'wb') as f:
        pickle.dump(userid_encoder, f)
    with open('temp/gametitle_encoder.pkl', 'wb') as f:
        pickle.dump(gametitle_encoder, f)
