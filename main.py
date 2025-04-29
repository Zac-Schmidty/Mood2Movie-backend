from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],
)

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
