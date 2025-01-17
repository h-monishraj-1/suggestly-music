import pickle
import requests
import pandas as pd

class MusicRecommender:
    def __init__(self, music_path, similarity_path):
        self.music_df = pickle.load(open(music_path, "rb"))
        self.similarity = pickle.load(open(similarity_path, "rb"))


    def recommend(self, music_title):
        try:
            music_index = self.music_df[self.music_df['title'].str.lower() == music_title.lower()].index[0]
            distances = self.similarity[music_index]
            music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

            recommended_songs = []
            for i in music_list:
                song_title = self.music_df.iloc[i[0]]['title']
                recommended_songs.append(song_title)
        
            return recommended_songs
        except IndexError:
            return [("Song not found. Please try another.")]
