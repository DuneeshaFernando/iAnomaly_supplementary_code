# We will use python:3.10-alpine as the base image for building the Flask container
FROM python:3.9-slim

# It specifies the working directory where the Docker container will run
WORKDIR /app

# Copying all the application files to the working directory
COPY . .

# Install all the dependencies required to run the Flask application
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Docker container for the application to run on port 4000
EXPOSE 5000

# The command required to run the Dockerized application
CMD ["python", "/app/if_ad_FLASK_LOCAL_RUN_STANDALONE.py"]