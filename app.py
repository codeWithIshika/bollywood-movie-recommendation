import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("bollywood_full.csv")

df["story"] = df["story"].fillna("")

df["clean_text"] = df["story"].str.lower()

df["clean_text"] = df["clean_text"].str.replace(
    r"[^a-zA-Z\s]", "", regex=True
)

stop_words = set(ENGLISH_STOP_WORDS)

df["clean_text"] = df["clean_text"].apply(
    lambda text: " ".join(
        word for word in text.split()
        if word not in stop_words
    )
)


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

similarity_matrix = cosine_similarity(tfidf_matrix)


def recommend(item_name, top_n=5):
    item_index = df[
        df["title_x"].str.lower() == item_name.lower()
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[item_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:top_n + 1]:
        recommendations.append(df.iloc[index]["title_x"])

    return recommendations

st.title("🎬 Bollywood Movie Recommendation System")

st.write("Select a movie to get similar movie recommendations.")

movie_name = st.selectbox(
    "Select a movie:",
    df["title_x"].tolist()
)

if st.button("Get Recommendations"):

    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("🎬", movie)