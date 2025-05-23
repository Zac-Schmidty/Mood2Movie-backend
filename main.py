import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import pickle
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import random
from difflib import get_close_matches
from functools import lru_cache
import time
import logging
from typing import Dict, Any, List
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specifically allow your frontend origin
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
    page: int = 1  # Default to page 1 if not specified

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, v):
        if not isinstance(v, str):
            raise ValueError("Mood must be a string")
        return v.lower()
    
    @field_validator('page')
    @classmethod
    def validate_page(cls, v):
        if v < 1:
            raise ValueError("Page number must be greater than 0")
        return v

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
    "family": {
        "genres": [10751, 16],  # Family, Animation
        "keywords": [10683, 158718, 180547],  # childhood, classic, heartwarming
        "sort_by": "vote_count.desc"
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
    "family": ["childish", "kids", "children", "family-friendly", "child-friendly"],
}

# Create a session for better connection pooling
session = None

@app.on_event("startup")
async def startup_event():
    global session
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        logger.info("Successfully created aiohttp session")
    except Exception as e:
        logger.error(f"Failed to create aiohttp session: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    if session:
        try:
            await session.close()
            logger.info("Successfully closed aiohttp session")
        except Exception as e:
            logger.error(f"Error closing aiohttp session: {str(e)}")

# Enhanced caching for movie details
@lru_cache(maxsize=1000)
def get_cached_movie_details(movie_id: int, api_key: str) -> dict:
    """Cache movie details to reduce API calls"""
    movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    response = requests.get(movie_details_url)
    return response.json()

async def fetch_movie_details(movie_id: int, api_key: str) -> dict:
    """Fetch movie details asynchronously"""
    if not session:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Backend service is not properly initialized"
            }
        )
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        async with session.get(url) as response:
            if response.status == 404:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Movie not found",
                        "message": f"No movie found with ID: {movie_id}"
                    }
                )
            response.raise_for_status()
            data = await response.json()
            
            # Validate the response data
            if not isinstance(data, dict):
                logger.error(f"Invalid response format for movie {movie_id}: {data}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Invalid response",
                        "message": "Received invalid data from movie service"
                    }
                )
            
            if 'id' not in data:
                logger.error(f"Missing movie ID in response for movie {movie_id}: {data}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Invalid response",
                        "message": "Received incomplete data from movie service"
                    }
                )
            
            return data
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching movie details: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Unable to connect to movie service"
            }
        )
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding movie details response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid response",
                "message": "Received invalid data from movie service"
            }
        )

async def fetch_credits(movie_id: int, api_key: str) -> dict:
    """Fetch movie credits asynchronously"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

async def fetch_videos(movie_id: int, api_key: str) -> dict:
    """Fetch movie videos asynchronously"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

async def fetch_release_dates(movie_id: int, api_key: str) -> dict:
    """Fetch movie release dates asynchronously"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

async def fetch_similar_movies(movie_id: int, api_key: str) -> dict:
    """Fetch similar movies asynchronously"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

async def fetch_actor_details(actor_id: int, api_key: str) -> dict:
    """Fetch actor details asynchronously"""
    url = f"https://api.themoviedb.org/3/person/{actor_id}?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

async def fetch_actor_movies(actor_id: int, api_key: str) -> dict:
    """Fetch actor's movies asynchronously"""
    url = f"https://api.themoviedb.org/3/person/{actor_id}/movie_credits?api_key={api_key}"
    async with session.get(url) as response:
        return await response.json()

# Custom exception handlers
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "message": str(exc.detail),
            "path": request.url.path
        }
    )

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "path": request.url.path
        }
    )

@app.exception_handler(503)
async def custom_503_handler(request: Request, exc: HTTPException):
    logger.error(f"Service unavailable: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service unavailable",
            "message": "Unable to connect to movie service. Please try again later.",
            "path": request.url.path
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test TMDB API connection
        test_url = f"https://api.themoviedb.org/3/movie/550?api_key={TMDB_API_KEY}"
        response = requests.get(test_url, timeout=5)
        response.raise_for_status()
        
        return {
            "status": "healthy",
            "tmdb_api": "connected",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Service dependencies unavailable",
                "message": str(e)
            }
        )

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.2f}s"
    )
    
    return response

@app.post("/recommendations/")
async def get_movie_recommendations(request: MoodRequest):
    try:
        input_mood = request.mood.lower().strip()
        current_page = request.page
        
        logger.info(f"Processing recommendation request for mood: {input_mood}, page: {current_page}")
        
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
                logger.warning(f"Invalid mood '{input_mood}' - suggesting: {suggestions}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Mood not found",
                        "message": f"Did you mean: {suggestions}? You can also try: {', '.join(MOOD_SYNONYMS.keys())}"
                    }
                )
            else:
                logger.warning(f"Invalid mood '{input_mood}' - no close matches found")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Unsupported mood",
                        "message": f"Try one of these moods: {', '.join(sorted(MOOD_SYNONYMS.keys()))}"
                    }
                )
        
        try:
            mood_data = MOOD_MAPPINGS[base_mood]
            genre_string = ",".join(map(str, mood_data["genres"]))
            
            # Add animation exclusion except for family/childish moods
            without_genres = "&without_genres=16" if base_mood not in ["family", "nostalgic"] else ""
            
            # First try with keywords
            discover_url = (
                f"https://api.themoviedb.org/3/discover/movie"
                f"?api_key={TMDB_API_KEY}"
                f"&with_genres={genre_string}"
                f"{without_genres}"
                f"&with_keywords={','.join(map(str, mood_data['keywords']))}"
                f"&sort_by={mood_data['sort_by']}"
                f"&vote_average.gte=7"
                f"&vote_count.gte=100"
                f"&language=en-US"
                f"&include_adult=false"
                f"&page={current_page}"
            )

            response = requests.get(discover_url)
            data = response.json()
            total_pages = data.get("total_pages", 1)
            all_movies = data.get("results", [])

            # If no results with keywords, try without keywords
            if not all_movies:
                discover_url = (
                    f"https://api.themoviedb.org/3/discover/movie"
                    f"?api_key={TMDB_API_KEY}"
                    f"&with_genres={genre_string}"
                    f"{without_genres}"
                    f"&sort_by={mood_data['sort_by']}"
                    f"&vote_average.gte=7"
                    f"&vote_count.gte=100"
                    f"&language=en-US"
                    f"&include_adult=false"
                    f"&page={current_page}"
                )
                response = requests.get(discover_url)
                data = response.json()
                total_pages = data.get("total_pages", 1)
                all_movies = data.get("results", [])

            # Filter movies
            filtered_movies = []
            for movie in all_movies:
                if movie['vote_average'] >= 7:
                    # For non-family/childish moods, check if movie is animated
                    if base_mood not in ["family", "nostalgic"]:
                        # Use cached movie details
                        movie_details = get_cached_movie_details(movie['id'], TMDB_API_KEY)
                        movie_genres = [genre['id'] for genre in movie_details.get('genres', [])]
                        
                        # Skip if movie is animated (genre 16)
                        if 16 in movie_genres:
                            continue
                    
                    filtered_movies.append(movie)

            # Handle cases where we don't have enough movies
            if not filtered_movies:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "No movies found",
                        "message": f"No movies found for mood: {base_mood}. Please try a different mood."
                    }
                )
            
            # If we have more than 12 movies, randomly select 12
            if len(filtered_movies) > 12:
                filtered_movies = random.sample(filtered_movies, 12)
            
            return {
                "recommendations": [
                    {
                        "id": movie["id"],
                        "title": movie["title"],
                        "overview": movie["overview"],
                        "rating": movie["vote_average"],
                        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                        "release_date": movie.get("release_date"),
                        "genres": [genre["name"] for genre in movie.get("genres", [])]
                    }
                    for movie in filtered_movies
                ],
                "total_pages": total_pages,
                "current_page": current_page
            }

        except requests.RequestException as e:
            logger.error(f"TMDB API request failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service unavailable",
                    "message": "Unable to connect to movie service. Please try again later."
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error in recommendations: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred. Please try again later."
                }
            )

    except requests.RequestException as e:
        logger.error(f"TMDB API request failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Unable to connect to movie service. Please try again later."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )

@app.get("/movie/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get detailed information about a specific movie"""
    try:
        logger.info(f"Fetching details for movie ID: {movie_id}")
        
        if not session:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service unavailable",
                    "message": "Backend service is not properly initialized"
                }
            )
        
        # Fetch all required data in parallel with timeout
        try:
            movie_details, credits, videos, release_dates, similar_movies = await asyncio.wait_for(
                asyncio.gather(
                    fetch_movie_details(movie_id, TMDB_API_KEY),
                    fetch_credits(movie_id, TMDB_API_KEY),
                    fetch_videos(movie_id, TMDB_API_KEY),
                    fetch_release_dates(movie_id, TMDB_API_KEY),
                    fetch_similar_movies(movie_id, TMDB_API_KEY)
                ),
                timeout=8.0  # 8 second timeout for all parallel requests
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching movie details for ID: {movie_id}")
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Gateway Timeout",
                    "message": "Request timed out while fetching movie details"
                }
            )
        
        # Validate movie details
        if not isinstance(movie_details, dict) or 'id' not in movie_details:
            logger.error(f"Invalid movie details response for ID {movie_id}: {movie_details}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Invalid response",
                    "message": "Received invalid data from movie service"
                }
            )
        
        # Check if current movie is animated
        is_current_movie_animated = any(genre['id'] == 16 for genre in movie_details.get('genres', []))
        
        # Process cast details asynchronously
        cast_details = []
        top_cast = credits.get("cast", [])[:5]  # Top 5 cast members
        
        # Fetch actor details and movies in parallel
        actor_tasks = []
        for cast in top_cast:
            if not isinstance(cast, dict) or 'id' not in cast:
                logger.warning(f"Invalid cast member data: {cast}")
                continue
            actor_tasks.extend([
                fetch_actor_details(cast['id'], TMDB_API_KEY),
                fetch_actor_movies(cast['id'], TMDB_API_KEY)
            ])
        
        if actor_tasks:
            actor_results = await asyncio.gather(*actor_tasks)
            
            # Process actor results
            for i in range(0, len(actor_results), 2):
                if i + 1 >= len(actor_results):
                    break
                    
                actor_data = actor_results[i]
                known_for_data = actor_results[i + 1]
                cast = top_cast[i // 2]
                
                if not isinstance(actor_data, dict) or not isinstance(known_for_data, dict):
                    logger.warning(f"Invalid actor data for cast member {cast.get('id')}")
                    continue
                
                # Get the highest quality profile image available
                profile_path = None
                if actor_data.get("profile_path"):
                    profile_path = f"https://image.tmdb.org/t/p/h632{actor_data['profile_path']}"
                
                # Get notable movies sorted by popularity (vote_count)
                notable_movies = []
                for movie in known_for_data.get("cast", []):
                    if not isinstance(movie, dict):
                        continue
                        
                    # Skip the current movie
                    if movie.get("id") == movie_id:
                        continue
                        
                    # Skip animations unless the current movie is animated
                    if not is_current_movie_animated:
                        if any(genre_id == 16 for genre_id in movie.get('genre_ids', [])):
                            continue
                    
                    # Include movies that have both a poster and meet minimum vote threshold
                    vote_count = movie.get("vote_count", 0)
                    vote_average = movie.get("vote_average", 0)
                    
                    # Minimum 1000 votes for credibility
                    if movie.get("poster_path") and vote_count >= 100:
                        notable_movies.append({
                            "id": movie["id"],
                            "title": movie["title"],
                            "poster_path": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                            "character": movie.get("character", ""),
                            "release_date": movie.get("release_date", ""),
                            "rating": round(vote_average, 1),
                            "vote_count": vote_count  # Include vote count for transparency
                        })
                
                # Sort movies by rating (descending) and take top 6
                notable_movies.sort(key=lambda x: x["rating"], reverse=True)
                notable_movies = notable_movies[:6]
                
                cast_details.append({
                    "id": cast["id"],
                    "name": cast["name"],
                    "character": cast["character"],
                    "profile_path": profile_path,
                    "biography": actor_data.get("biography", ""),
                    "birthday": actor_data.get("birthday"),
                    "place_of_birth": actor_data.get("place_of_birth"),
                    "known_for_department": actor_data.get("known_for_department"),
                    "notable_movies": notable_movies
                })
        
        # Get content rating (US rating if available, otherwise first available)
        content_rating = None
        for release in release_dates.get("results", []):
            if release["iso_3166_1"] == "US":
                for rating in release.get("release_dates", []):
                    if rating.get("certification"):
                        content_rating = rating["certification"]
                        break
                break
        if not content_rating:
            for release in release_dates.get("results", []):
                for rating in release.get("release_dates", []):
                    if rating.get("certification"):
                        content_rating = rating["certification"]
                        break
                if content_rating:
                    break
        
        # Get director and writer information
        director = None
        writers = []
        for crew in credits.get("crew", []):
            if crew["job"] == "Director":
                director = {
                    "name": crew["name"],
                    "profile_path": f"https://image.tmdb.org/t/p/w185{crew['profile_path']}" if crew.get("profile_path") else None
                }
            elif crew["job"] in ["Screenplay", "Writer", "Story"]:
                writers.append({
                    "name": crew["name"],
                    "job": crew["job"],
                    "profile_path": f"https://image.tmdb.org/t/p/w185{crew['profile_path']}" if crew.get("profile_path") else None
                })
        
        # Get official trailer and teaser
        trailer = None
        teaser = None
        for video in videos.get("results", []):
            if video["site"] == "YouTube":
                if video["type"] == "Trailer" and not trailer:
                    trailer = {
                        "name": video["name"],
                        "key": video["key"],
                        "type": video["type"]
                    }
                elif video["type"] == "Teaser" and not teaser:
                    teaser = {
                        "name": video["name"],
                        "key": video["key"],
                        "type": video["type"]
                    }
        
        # Format the response
        return {
            "id": movie_details.get("id"),
            "title": movie_details.get("title"),
            "overview": movie_details.get("overview"),
            "poster_path": f"https://image.tmdb.org/t/p/w500{movie_details.get('poster_path')}" if movie_details.get("poster_path") else None,
            "backdrop_path": f"https://image.tmdb.org/t/p/original{movie_details.get('backdrop_path')}" if movie_details.get("backdrop_path") else None,
            "release_date": movie_details.get("release_date"),
            "runtime": movie_details.get("runtime"),
            "rating": movie_details.get("vote_average"),
            "vote_count": movie_details.get("vote_count"),
            "content_rating": content_rating,
            "genres": [genre["name"] for genre in movie_details.get("genres", [])],
            "cast": cast_details,
            "director": director,
            "writers": writers,
            "trailer": trailer,
            "teaser": teaser,
            "similar_movies": [
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "poster_path": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                    "rating": movie["vote_average"]
                }
                for movie in similar_movies.get("results", [])[:6]
            ]
        }
    except aiohttp.ClientError as e:
        logger.error(f"TMDB API request failed for movie {movie_id}: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Unable to connect to movie service. Please try again later."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching movie {movie_id}: {str(e)}")
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
