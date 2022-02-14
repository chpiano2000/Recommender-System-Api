import pickle
import pandas as pd
import numpy as np

def predict_model(user_id, loaded_model):
    # loaded_model = pickle.load(open('recommender.pkl', 'rb'))
    movie_df = pd.read_csv('ml-latest-small/movies.csv')
    df = pd.read_csv("ml-latest-small/ratings.csv")
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    # userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[~movie_df['movieId'].isin(movies_watched_by_user.movieId.values)]['movieId']
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = loaded_model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]
    top_movies_user = (
            movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
        )

    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]

    recommended_movies_list = []
    movie_df_rows_list = []
    for row in recommended_movies.itertuples():
        recommended_movies_list.append({
            "title": row.title, 
            "genres": row.genres, 
            "movieId": row.movieId
        })

    for row in movie_df_rows.itertuples():
        previous = df.loc[(df.userId == user_id) & (df.movieId == row.movieId)]
        movie_df_rows_list.append({
            "title": row.title, 
            "genres": row.genres, 
            "movieId": row.movieId,
            "rating": previous.rating.values[0].tolist(),
            "timestamp": previous.timestamp.values[0].tolist()
        })

    return recommended_movies_list, movie_df_rows_list
    # return recommended_movies, movie_df_rows

