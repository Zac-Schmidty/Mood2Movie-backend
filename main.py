from fastapi import FastAPI
from pydantic import BaseModel
from model import recommend_movies
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoodInput(BaseModel):
    mood: str

@app.post("/recommend")
def get_recommendations(input: MoodInput):
    return {"movies": recommend_movies(input.mood)}
