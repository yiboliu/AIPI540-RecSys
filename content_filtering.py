import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def create_sim_matrix(game_features):
    """This function creates the similarity matrix, calculated with cosine similarity.
    Two features, popularity and average play time are scaled before being used to calculate similarity"""
    features = np.column_stack((game_features['popularity'], game_features['avg_play_time']))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    sim_matrix = cosine_similarity(features)
    sim_matrix = pd.DataFrame(sim_matrix, index=game_features.index, columns=game_features.index)
    return sim_matrix


def generate_recommendations_content(userId):
    """Generate the recommendation for the user based on content filtering.
    Top 10 games predicted to be played the longest are returned."""
    data = pd.read_csv('temp/data.csv')
    game_features = pd.read_csv('temp/game_features.csv')
    with open('temp/gametitle_encoder.pkl', 'rb') as f:
        gametitle_encoder = pickle.load(f)
    with open('temp/userid_encoder.pkl', 'rb') as f:
        userid_encoder = pickle.load(f)
    userIdInt = userid_encoder.transform([userId])[0]
    simtable = create_sim_matrix(game_features)
    user_hours = data.loc[data['userIdInt'] == userIdInt]
    user_hours = user_hours.sort_values(by='hours', axis=0, ascending=False)
    topGameTitleInt = user_hours.iloc[0, :]['gameTitleInt']
    sims = simtable.loc[topGameTitleInt, :]
    sims = sims.sort_values(ascending=False)
    sims = sims.drop(topGameTitleInt)
    games = []
    for i in range(10):
        games.append(gametitle_encoder.inverse_transform([sims.index[i]])[0])
    return games
