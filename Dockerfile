# Use the official Python base image with version 3.9
FROM python:3.9

WORKDIR /app

# Copy the requirements file into the container at /app
COPY web_service/requirements.txt .

# Install packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY web_service/deploy.py /app/

# Expose the port the server will run on
EXPOSE 5000

# Define environment variables
ENV MLFLOW_TRACKING_URI http://localhost:5000

CMD ["python", "deploy.py"]
