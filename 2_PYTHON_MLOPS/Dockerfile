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

# Command to run the ZMQ Inference Server when the container starts
# This exposes the ZMQ port 5555
EXPOSE 5555
CMD ["python", "src/inference_server.py"]