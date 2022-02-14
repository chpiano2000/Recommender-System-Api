from utils import predict_model
from fastapi import FastAPI
import pickle

app = FastAPI()

load_model = pickle.load(open('recommender.pkl', 'rb'))

@app.post("/history/{user_id}")
def predict(user_id: int):
    _ , history = predict_model(user_id, load_model)
    return history

@app.post("/recommender/{user_id}")
def recommend(user_id: int):
    recommended_movies, _ = predict_model(user_id, load_model)
    return recommended_movies
# recomender, history = predict_model(3)