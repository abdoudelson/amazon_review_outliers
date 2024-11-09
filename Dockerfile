# my_ml_project/Dockerfile

# Use official Python image as the base
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose MLflow tracking server port and FastAPI port
EXPOSE 5000 8000

# Run both MLflow server and FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
