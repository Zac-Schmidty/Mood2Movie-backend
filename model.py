def recommend_movies(mood):
    mood_dict = {
        "happy": ["Paddington", "Sing Street", "Up"],
        "sad": ["Inside Out", "Blue Valentine", "The Pursuit of Happyness"],
        "angry": ["Gladiator", "John Wick", "Falling Down"],
        "romantic": ["The Notebook", "Pride & Prejudice", "La La Land"]
    }
    return mood_dict.get(mood.lower(), ["No suggestions found. Try another mood."])
