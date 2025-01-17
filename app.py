from flask import Flask, request, jsonify, render_template
from music_recommendation import MusicRecommender
import pandas as pd

app = Flask(__name__)

# Initialize the music recommender
music_recommender = MusicRecommender("music_list.pkl", "similarity.pkl")

# Load the music dataframe globally
music_df = pd.read_pickle("music_list.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend_music", methods=["POST"])
def recommend_music():
    data = request.json
    music_title = data.get("music_title", "").strip()
    if not music_title:
        return jsonify({"recommendations": []})
    
    recommendations = music_recommender.recommend(music_title)
    return jsonify({"recommendations": [{"title": rec.title()} for rec in recommendations]})

@app.route("/get_titles", methods=["GET"])
def get_titles():
    query = request.args.get("q", "").lower().strip()
    if not query:
        return jsonify({"titles": []})

    try:
        matching_titles = (
            music_df[music_df['title'].str.lower().str.contains(query, case=False, na=False)]['title']
            .head(10)
            .str.title()
            .tolist()
        )

        return jsonify({"titles": matching_titles})
    except KeyError:
        return jsonify({"titles": []})

if __name__ == "__main__":
    app.run(debug=True)
