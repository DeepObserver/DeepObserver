FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: We don't copy code here because we use volume mounting in development
# COPY . .

# Command to run the application
CMD ["uvicorn", "deepobserver.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]