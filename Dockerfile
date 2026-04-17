# Use an official Python runtime
FROM python:3.10-slim

# Set the working directory
WORKDIR /main

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the specific port Hugging Face requires
EXPOSE 7860

# Boot up Uvicorn, pointing to port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]