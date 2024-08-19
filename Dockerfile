# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV FLASK_APP=deploy.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY web_service/ /app/

# Expose the port Flask will run on
EXPOSE 8080

# Use a non-root user to improve security
RUN useradd -m flaskuser
USER flaskuser

# Define the entry point to run the Flask app
ENTRYPOINT ["python", "deploy.py"]

# Define the default command (can be overridden)
CMD ["deploy.py"]
