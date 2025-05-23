#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Start the FastAPI backend with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000 