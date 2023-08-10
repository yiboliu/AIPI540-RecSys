import torch
import pandas as pd
import pickle


def predict_hours(model, userIdInt, gameTitleInt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([userIdInt, gameTitleInt]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred


def generate_recommendations_collab(userId):
    data = pd.read_csv('temp/data.csv')
    purchased = pd.read_csv('temp/purchased.csv')
    model = torch.load('models/collab.pth')
    model.eval()
    with open('temp/gametitle_encoder.pkl', 'rb') as f:
        gametitle_encoder = pickle.load(f)
    with open('temp/userid_encoder.pkl', 'rb') as f:
        userid_encoder = pickle.load(f)
    preds = {}
    visited = set()
    games_purchased = purchased.loc[purchased['userId'] == userId, 'gameTitle'].tolist()
    for game in data['gameTitle'].tolist():
        if game in games_purchased or game in visited:
            continue
        visited.add(game)
        userIdInt = userid_encoder.transform([userId])[0]
        gameTitleInt = gametitle_encoder.transform([game])[0]
        pred = predict_hours(model, userIdInt, gameTitleInt)
        preds[game] = pred.detach().cpu().item()

    recs = sorted(preds.items(), key=lambda item: item[1], reverse=True)[:10]
    return [rec[0] for rec in recs]
