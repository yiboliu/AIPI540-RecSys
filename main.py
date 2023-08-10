import os
import streamlit as st

from serving_model import generate_recommendations_collab
from content_filtering import generate_recommendations_content


def generate_recs(userId):
    """This function generates recommendations with two approaches, collaborative and content filtering with the given
    user id. Note: as the user id is given as a string, this function converts it to an int for encoding/decoding
    purpose."""
    userId = int(userId)
    collab_res = generate_recommendations_collab(userId=userId)
    content_res = generate_recommendations_content(userId=userId)
    return collab_res, content_res


def main():
    """This function launches the frontend to let users interact with"""
    if not os.path.exists('temp/data.csv') or not os.path.exists('temp/purchased.csv') \
            or not os.path.exists('temp/gametitle_encoder.pkl') or not os.path.exists('temp/userid_encoder.pkl') \
            or not os.path.exists('models/collab.pth') or not os.path.exists('temp/game_features.csv'):
        raise RuntimeError('Have you ran setup.py to create temp/ and required files?')
    st.title('Game Recommender')
    userId = st.text_input("Please enter a valid user id: ")

    if st.button('Recommend'):
        if userId:
            print('yes')
            collab_res, content_res = generate_recs(userId)
            st.write(f'Recommendation results from collaborative filtering: {collab_res}')
            st.write(f'Recommendation results from content filtering: {content_res}')


if __name__ == "__main__":
    main()
