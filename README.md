# Movie-Recommendation-System
The Emotion-Based Movie Recommendation System is an AI chatbot that detects a user’s mood using NLP and suggests movies that match their emotions. Built with Python, Transformers, Pandas, and Gradio, it provides an interactive, user-friendly experience and runs easily on Google Colab for entertainment and analysis.
# 🎬 Emotion-Based Movie Recommendation Chatbot
# -------------------------------------------------------
# This chatbot detects user emotion and recommends movies
# based on their mood using an NLP model and Gradio UI.
# -------------------------------------------------------

# ✅ Step 1: Install & Import Required Libraries 
!pip install transformers gradio torch pandas --quiet

import pandas as pd
import torch
from transformers import pipeline
import gradio as gr

# ✅ Step 2: Load Emotion Detection Model
emotion_analyzer = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

# ✅ Step 3: Create / Load Movie Dataset
# You can replace this with your own dataset if desired.
data = {
    "mood": ["joy", "sadness", "anger", "fear", "love", "surprise", "neutral"],
    "movies": [
        ["The Intern", "Paddington 2", "The Secret Life of Walter Mitty", "Inside Out"],
        ["The Pursuit of Happyness", "A Monster Calls", "Hachi: A Dog's Tale", "The Green Mile"],
        ["John Wick", "Gladiator", "The Dark Knight", "Mad Max: Fury Road"],
        ["Get Out", "A Quiet Place", "The Sixth Sense", "Bird Box"],
        ["Pride & Prejudice", "The Notebook", "La La Land", "Titanic"],
        ["Inception", "Now You See Me", "Edge of Tomorrow", "The Prestige"],
        ["Forrest Gump", "The Shawshank Redemption", "The Social Network", "Moneyball"]
    ]
}

movies_df = pd.DataFrame(data)
movies_df.to_csv("/content/movies_dataset.csv", index=False)

print("✅ Movie dataset created successfully!")
print(movies_df.head())

# ✅ Step 4: Define Mood-to-Movie Mapping
mood_to_movie = {
    "joy": ["The Intern", "Paddington 2", "The Secret Life of Walter Mitty", "Inside Out"],
    "sadness": ["The Pursuit of Happyness", "A Monster Calls", "Hachi: A Dog's Tale", "The Green Mile"],
    "anger": ["John Wick", "Gladiator", "The Dark Knight", "Mad Max: Fury Road"],
    "fear": ["Get Out", "A Quiet Place", "The Sixth Sense", "Bird Box"],
    "love": ["Pride & Prejudice", "The Notebook", "La La Land", "Titanic"],
    "surprise": ["Inception", "Now You See Me", "Edge of Tomorrow", "The Prestige"],
    "neutral": ["Forrest Gump", "The Shawshank Redemption", "The Social Network", "Moneyball"]
}

# ✅ Step 5: Chatbot Function
def chatbot_response(message, chat_history=[]):
    # Detect the emotion from user input
    emotion = emotion_analyzer(message)[0]['label'].lower()

    # Get recommended movies for that mood
    movies = mood_to_movie.get(emotion, mood_to_movie["neutral"])

    # Create chatbot reply
    response = f"🎭 You seem to be feeling **{emotion}**.\nHere are some movies that might match your mood:\n"
    for m in movies:
        response += f"🎬 {m}\n"

    chat_history.append((message, response))
    return "", chat_history

# ✅ Step 6: Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Welcome to Emotion-Based AI Movie Recommender 🍿")
    gr.Markdown("Hey there! 👋 Tell me how you feel, and I’ll suggest some great movies to match your mood.")

    chatbot = gr.Chatbot(label="🎥 Movie Mood Assistant")
    msg = gr.Textbox(placeholder="Type how you feel...", label="Your Mood 💬")
    clear = gr.Button("Clear Chat")

    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
