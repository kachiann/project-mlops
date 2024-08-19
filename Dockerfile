# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV FLASK_APP=deploy.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000


# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files
COPY web_service/ /app/

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 8080

# Define the command to run the app
CMD ["python", "deploy.py"]
