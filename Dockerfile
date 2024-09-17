# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download OpenCV pre-trained model files for face detection
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

# Set the environment variable to not use buffering (for easier container logs)
ENV PYTHONUNBUFFERED=1

# Expose the port that Flask will run on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
