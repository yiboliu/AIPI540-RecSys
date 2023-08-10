# AIPI540-RecSys
This is the Recommendation System for course AIPI 540 at Duke University

## Overview
This project serves as a recommender of games for steam users. The dataset provided included user_id's, game titles 
(equivalence to id), whether the user purchased that game, and hours of playing that game. 
The primary metrics we use to determine which game to recommend is the predicted hours of playing. 
The longer the user is expected to play, the more likely the user is going to like it. 
We also excluded the games that has been purchased by the user from the recommendation list.
This is important to the business of steam as this will improve the user experience of customers and potentially raise 
the likelihood of purchasing, which contributes to income.

## Source of Data
The data comes from this Kaggle competition: https://www.kaggle.com/datasets/tamber/steam-video-games

## Uniqueness
I checked a number of solutions posted, and found none of them using collaborative filtering, and thus I regard usage of
collaborative filtering as an innovative solution. 

Another point I regard as more innovative is that I created a few new features for content filtering. 
I calculated the average play time and popularity (the number of users purchased) as features for games and used them to
perform content filtering. 

## Solutions
The first solution is collaborative filtering using Deep Learning model. The model learns from embeddings of playing 
hours distributions of games for each user and predicts the hours for any given user.

The second solution is content filtering, using average play time and popularity of games as features.

## Setup

To set up the project, just install all the dependencies in the requirements.txt, by running ``pip install -r requirements.txt`` 
and run ``python setup.py`` to generate temp/ path and required intermediate files.

To train the deep learning model, run ``python modeling.py``. This will train the model and save the related artifacts 
in `models/` directory for later use.

Content filtering solution doesn't involve any modeling but works real time. When you click the UI button to call the method, 
it will run in the background to render the result.
## Demo

To have a try with the demo, run ``streamlit run main.py --server.port=8080 --server.address=0.0.0.0``. 
(Please bear for a few seconds for it to fully start) 
You will see a prompt for entering an user id, and you will see the results from both DL and non DL solutions.

