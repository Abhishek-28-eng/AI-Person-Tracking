# Use the latest Python base image 
FROM python:3-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy tracking code and model weights
COPY track/ /app/track/

# Copy sources file
COPY sources.txt /app/

# Set the working directory to /app/track so paths match your VS Code setup
WORKDIR /app/track

# Command to run the tracker
ENTRYPOINT ["python3", "track.py"]
