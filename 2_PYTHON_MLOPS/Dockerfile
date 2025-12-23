# Base image for a stable Python environment
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Using --no-cache-dir reduces image size
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libta-lib-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the core application code
COPY src/ ./src/
COPY config/ ./config/

# Command to run the FastAPI Inference Server when the container starts
# Expose port 8000 for HTTP API
EXPOSE 8000
CMD ["uvicorn", "src.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]