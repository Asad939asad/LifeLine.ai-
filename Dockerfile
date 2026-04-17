# Use an official Python runtime
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# --- FIX START: Install modern system dependencies for Trixie ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
# --- FIX END ---

# Copy your requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port
EXPOSE 7860

# Boot up Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]