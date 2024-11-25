import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# load and cache the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/song_dataset.csv')
        song_titles = pd.Series(df.title.values, index=df.song).to_dict()
        return df, song_titles
    except FileNotFoundError:
        st.error("Error: 'song_dataset.csv' not found. Make sure the file is in the correct location.")
        return None, None

#  user-song matrix and normalise it 
def create_user_song_matrix(df):
    user_song = df.pivot_table(index='user', columns='song', values='play_count', aggfunc='sum', fill_value=0)
    scaler = MinMaxScaler()
    user_song_norm = pd.DataFrame(scaler.fit_transform(user_song), index=user_song.index, columns=user_song.columns)
    return user_song, user_song_norm

# similarity between users
def calc_user_similarity(user_song_norm):
    return cosine_similarity(user_song_norm)

# popular songs recommendation
def get_popular_songs(user_song, song_titles, num_songs=5):
    top_songs = user_song.sum().nlargest(num_songs)
    return [(song_titles.get(song, song), score) for song, score in top_songs.items()]

# generate recommendation for songs 
def recommend_songs(user_id, listened_songs, user_song, user_similarity, song_titles, num_songs=5):
    if user_id not in user_song.index:
        return get_popular_songs(user_song, song_titles, num_songs)
    
    user_index = list(user_song.index).index(user_id)
    similar_users = np.argsort(user_similarity[user_index])[::-1][1:]
    
    user_songs_set = set(listened_songs)
    scores = {}
    for similar_user_index in similar_users:
        similar_user = user_song.index[similar_user_index]
        sim_score = user_similarity[user_index][similar_user_index]
        similar_user_songs = user_song.columns[user_song.loc[similar_user] > 0]
        
        for song in similar_user_songs:
            if song not in user_songs_set:
                score = sim_score * user_song.loc[similar_user, song]
                scores[song] = scores.get(song, 0) + score

    if len(scores) < num_songs:
        top_songs = get_popular_songs(user_song, song_titles, num_songs)
        for song, score in top_songs:
            if song not in scores and song not in user_songs_set:
                scores[song] = score
    
    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_songs]
    return [(song_titles.get(song, song), score) for song, score in sorted_recs]

# WebApp
def main():
    st.title("Music Recommendation System")
    
    data, song_titles = load_data()
    if data is None:
        return
    
    user_song, user_song_norm = create_user_song_matrix(data)
    user_similarity = calc_user_similarity(user_song_norm)
    
    tab_existing, tab_new = st.tabs(["Existing User", "New User"])
    
    with tab_existing:
        st.subheader("Recommendations for Existing Users")
        user_id = st.selectbox("Select your user ID:", user_song.index)
        user_songs = user_song.columns[user_song.loc[user_id] > 0].tolist()
        
        with st.expander("Your Listening History"):
            st.write("Songs you've already listened to:")
            for song_id in user_songs[:10]:
                song_title = song_titles.get(song_id, song_id)
                st.write(f"- {song_title}")
            if len(user_songs) > 10:
                st.write(f"...and {len(user_songs) - 10} more songs")
    
    with tab_new:
        st.subheader("Recommendations for New Users")
        selected_songs = st.multiselect(
            "Select songs you've listened to:",
            options=[(song_id, title) for song_id, title in song_titles.items()],
            format_func=lambda x: x[1]
        )
        user_songs = [song_id for song_id, _ in selected_songs]
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding your recommendations..."):
            recommendations = recommend_songs(
                user_id if "user_id" in locals() else None,
                user_songs,
                user_song,
                user_similarity,
                song_titles
            )
            
            st.subheader("Recommended Songs")
            for title, score in recommendations:
                score_percentage = min(100, score * 100) if score <= 1 else score
                st.write(f"🎵 {title} (Match: {score_percentage:.1f}%)")

if __name__ == "__main__":
    main()
