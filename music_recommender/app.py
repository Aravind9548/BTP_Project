from flask import Flask, render_template, request
import pandas as pd
from utils.recommend import recommend_songs

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    mood = request.form.get('mood')
    genre = request.form.get('genre')
    text_input = request.form.get('text_input')

    recommendations = recommend_songs(user_mood=mood, genre=genre, user_text=text_input)

    return render_template('results.html', mood=mood, genre=genre, text_input=text_input, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
