from utils import predict_model
from fastapi import FastAPI, HTTPException, Depends
import pickle
import pandas as pd
import jwt
from datetime import datetime, timedelta
from typing import Union, Any
from pydantic import BaseModel
from security import validate_token, reusable_oauth2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_model = pickle.load(open('recommender.pkl', 'rb'))
user_data = pd.read_csv("ml-latest-small/ratings.csv")["userId"].unique().tolist()
print(user_data)

SECURITY_ALGORITHM = 'HS256'
SECRET_KEY = '123456'

class Login(BaseModel):
    username: int
    password: str="password"

def generate_token(username: Union[str, Any]) -> str:
    expire = datetime.utcnow() + timedelta(
        seconds=60 * 60 * 24 * 3  # Expired after 3 days
    )
    to_encode = {
        "exp": expire, "username": username
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=SECURITY_ALGORITHM)
    return encoded_jwt

# @app.post("/history/{user_id}", dependencies=[Depends(reusable_oauth2)])
@app.post("/history/{user_id}")
def predict(user_id: int):
    _ , history = predict_model(user_id, load_model)
    return history

# @app.post("/recommender/{user_id}", dependencies=[Depends(reusable_oauth2)])
@app.post("/recommender/{user_id}")
def recommend(user_id: int):
    recommended_movies, _ = predict_model(user_id, load_model)
    return recommended_movies

@app.post("/login")
def login(login: Login):
    if login.username in user_data:
        if login.password == "password":
            token = generate_token(login.username)
            return {"token": token}
        else:
            raise HTTPException(status_code=404, detail="password invalid")
    else:
        raise HTTPException(status_code=404, detail="User not found")