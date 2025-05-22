import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pickle
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import random
from difflib import get_close_matches

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],
)

load_dotenv()  # Load environment variables from .env file

# Replace the hardcoded API key with environment variable
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY environment variable is not set")

class MoodRequest(BaseModel):
    mood: str

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, v):
        if not isinstance(v, str):
            raise ValueError("Mood must be a string")
        return v.lower()

# Mood to genre/keyword mapping with TMDb IDs
MOOD_MAPPINGS = {
    "happy": {
        "genres": [35],  # Comedy only
        "keywords": [9717, 9716, 180547],  # feel-good, uplifting, cheerful
        "sort_by": "vote_count.desc"
    },
    "nostalgic": {
        "genres": [10751, 16],  # Family, Animation (moved from happy)
        "keywords": [10683, 158718, 180547],  # childhood, classic, heartwarming
        "sort_by": "vote_count.desc"
    },
    "sad": {
        "genres": [18],  # Drama
        "keywords": [9840, 161916, 12392],  # emotional, touching, melancholy
        "sort_by": "vote_average.desc"
    },
    "excited": {
        "genres": [28, 12],  # Action, Adventure
        "keywords": [9748, 9725, 163079],  # thrilling, adventure, action-packed
        "sort_by": "primary_release_date.desc"
    },
    "relaxed": {
        "genres": [10749, 35],  # Romance, Comedy
        "keywords": [9713, 171891, 9927],  # light-hearted, calm, peaceful
        "sort_by": "vote_count.desc"
    },
    "inspired": {
        "genres": [18, 36],  # Drama, History
        "keywords": [9672, 161902, 10683],  # inspiring, triumph, achievement
        "sort_by": "vote_average.desc"
    },
    "adventurous": {
        "genres": [12, 14],  # Adventure, Fantasy
        "keywords": [9725, 9748, 10683],  # adventure, journey, discovery
        "sort_by": "vote_count.desc"
    },
    "mysterious": {
        "genres": [9648, 53],  # Mystery, Thriller
        "keywords": [10714, 163079, 10361],  # suspense, twist, investigation
        "sort_by": "vote_average.desc"
    },
    "romantic": {
        "genres": [10749],  # Romance
        "keywords": [9663, 156209, 161335],  # love, romance, relationship
        "sort_by": "vote_count.desc"
    },
    "thoughtful": {
        "genres": [18, 878],  # Drama, Science Fiction
        "keywords": [9673, 163079, 10683],  # thought-provoking, philosophical, deep
        "sort_by": "vote_average.desc"
    },
    "angry": {
        "genres": [28, 53],  # Action, Thriller
        "keywords": [9748, 163079, 9725],  # intense, action-packed, adrenaline
        "sort_by": "vote_count.desc"
    },
    "bored": {
        "genres": [99, 36],  # Documentary, History
        "keywords": [9672, 161902, 180547],  # educational, fascinating, enlightening
        "sort_by": "vote_average.desc"  # Prioritize highly-rated docs
    },
}

# Mood synonyms mapping
MOOD_SYNONYMS = {
    "sad": ["unhappy", "depressed", "gloomy", "melancholy", "down", "blue"],
    "happy": ["joyful", "cheerful", "upbeat", "pleased", "content", "delighted", "funny", "amused"],
    "excited": ["thrilled", "energetic", "pumped", "eager", "enthusiastic"],
    "relaxed": ["calm", "peaceful", "chill", "mellow", "tranquil", "zen"],
    "nostalgic": ["sentimental", "reminiscent", "retro", "classic", "childhood", "family", "childish"],
    "inspired": ["motivated", "uplifted", "encouraged", "determined", "empowered"],
    "adventurous": ["daring", "bold", "brave", "exploring", "venturous"],
    "mysterious": ["intrigued", "curious", "suspenseful", "puzzled"],
    "romantic": ["loving", "affectionate", "passionate", "tender"],
    "thoughtful": ["reflective", "contemplative", "philosophical", "deep"],
    "angry": ["mad", "furious", "enraged", "irritated", "annoyed", "frustrated"],
    "bored": ["restless", "uninterested", "unstimulated", "idle", "indifferent", "looking to learn"],
}

@app.post("/recommendations/")
async def get_movie_recommendations(request: MoodRequest):
    input_mood = request.mood.lower().strip()
    
    # Create a list of all valid moods and synonyms
    valid_moods = []
    mood_to_base = {}  # Map to track which base mood each word belongs to
    for base_mood, synonyms in MOOD_SYNONYMS.items():
        valid_moods.append(base_mood)
        mood_to_base[base_mood] = base_mood
        for synonym in synonyms:
            valid_moods.append(synonym)
            mood_to_base[synonym] = base_mood

    # Find the base mood from exact matches first
    base_mood = None
    for mood, synonyms in MOOD_SYNONYMS.items():
        if input_mood == mood or input_mood in synonyms:
            base_mood = mood
            break
    
    if not base_mood:
        # Try to find close matches
        close_matches = get_close_matches(input_mood, valid_moods, n=3, cutoff=0.6)
        
        if close_matches:
            # Get the base moods for these matches
            suggested_moods = []
            for match in close_matches:
                base_mood = mood_to_base[match]
                if base_mood not in suggested_moods:
                    suggested_moods.append(base_mood)
            
            suggestions = ", ".join(suggested_moods)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Mood not found",
                    "message": f"Did you mean: {suggestions}? You can also try: {', '.join(MOOD_SYNONYMS.keys())}"
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported mood",
                    "message": f"Try one of these moods: {', '.join(sorted(MOOD_SYNONYMS.keys()))}"
                }
            )
    
    try:
        mood_data = MOOD_MAPPINGS[base_mood]
        genre_string = ",".join(map(str, mood_data["genres"]))
        
        # Randomly select a page between 1 and 3
        random_page = random.randint(1, 3)
        
        # Add animation exclusion except for nostalgic and adventurous moods
        without_genres = "&without_genres=16,10751" if base_mood not in ["nostalgic", "adventurous"] else ""
        
        # First try with both genres and keywords
        discover_url = (
            f"https://api.themoviedb.org/3/discover/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&with_genres={genre_string}"
            f"{without_genres}"  # Exclude both animation (16) and family (10751)
            f"&with_keywords={','.join(map(str, mood_data['keywords']))}"
            f"&sort_by={mood_data['sort_by']}"
            f"&vote_average.gte=7"
            f"&vote_count.gte=100"
            f"&language=en-US"
            f"&include_adult=false"
            f"&page={random_page}"
        )

        response = requests.get(discover_url)
        data = response.json()
        movies = data.get("results", [])

        # If no results, try again with just genres
        if not movies:
            discover_url = (
                f"https://api.themoviedb.org/3/discover/movie"
                f"?api_key={TMDB_API_KEY}"
                f"&with_genres={genre_string}"
                f"&sort_by={mood_data['sort_by']}"
                f"&vote_average.gte=7"
                f"&vote_count.gte=100"
                f"&language=en-US"
                f"&include_adult=false"
                f"&page={random_page}"
            )
            response = requests.get(discover_url)
            data = response.json()
            movies = data.get("results", [])

        # Get more movies than needed and randomly select from them
        filtered_movies = [
            movie for movie in movies 
            if movie['vote_average'] >= 7
        ]
        
        # If we have more than 12 movies, randomly select 12
        if len(filtered_movies) > 12:
            filtered_movies = random.sample(filtered_movies, 12)
        
        return {
            "recommendations": [
                {
                    "title": movie["title"],
                    "overview": movie["overview"],
                    "rating": movie["vote_average"],
                    "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                }
                for movie in filtered_movies
            ]
        }

    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Unable to connect to movie service. Please try again later."
            }
        )
    except Exception as e:
        # Log the unexpected error here if you have logging set up
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )

# ML modeling code
"""
# Load model
with open("movie_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

class MoodRequest(BaseModel):
    mood: str

@app.post("/predict")
async def predict_movies(request: MoodRequest):
    try:
        mood_vec = vectorizer.transform([request.mood])
        movies = model.predict(mood_vec)[0]
        return {"movies": movies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
