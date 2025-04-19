# Use a slim Python base image
FROM python:3.11-slim

# Install system dependencies (for Tesseract, PyMuPDF, and images)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmupdf-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 10000

# Run the app using Gunicorn
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:10000"]
