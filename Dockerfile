# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
# Hugging Face Spaces requires the app to run on port 7860
EXPOSE 7860

# Run uvicorn when the container launches
# --host 0.0.0.0 is required for external access
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
