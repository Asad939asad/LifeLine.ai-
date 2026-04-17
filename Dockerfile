# Use an official Python runtime
FROM python:3.10-slim

# Set the working directory (Standard convention)
WORKDIR /app

# --- FIX START: Install minimal system dependencies for OpenCV/YOLO ---
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
# --- FIX END ---

# Copy your requirements first (this makes builds faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code into /app
COPY . .

# Expose the specific port Hugging Face requires
EXPOSE 7860

# Boot up Uvicorn
# main:app means "Look in main.py for the object named app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]