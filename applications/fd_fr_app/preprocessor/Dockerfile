# We will use python:3.10-alpine as the base image for building the Flask container
FROM python:3.9-slim

# It specifies the working directory where the Docker container will run
WORKDIR /app

# Copying all the application files to the working directory
COPY . .

# Install all the dependencies required to run the Flask application
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# Make the data directory
RUN mkdir /data

COPY gray_image.npy /data/gray_image.npy

# Expose the Docker container for the application to run on port 3000
EXPOSE 3000

# The command required to run the Dockerized application
CMD ["python", "/app/preprocessing_app_STANDALONE_with_single_img_input.py"]