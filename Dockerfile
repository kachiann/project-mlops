# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Command to run the Flask application
CMD ["python", "web_service/deploy.py"]
