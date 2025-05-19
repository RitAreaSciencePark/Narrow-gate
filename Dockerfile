# Base image with CUDA support (if you need GPU acceleration)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Alternatively, for CPU-only support, you can use:
# FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        git \
        build-essential \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install torch with CUDA support (adjust the version and CUDA as needed)
RUN pip install torch==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY . .

# Expose the port for Streamlit (if used)
EXPOSE 8501

# Set the default command to run your application (adjust as needed)
CMD ["streamlit", "run", "your_main_script.py"]
