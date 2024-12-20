# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements_dev.txt .
# COPY requirements.txt .

# Install req libs

ENV PYHTONUNBUFFERED=1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr -y

# Install the required packages
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_dev.txt
RUN python -m nltk.downloader punkt averaged_perceptron_tagger
# Copy the project files to the working directory
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py","--theme.base","light"]
