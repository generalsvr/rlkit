FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt
COPY requirements.extra.txt /app/requirements.extra.txt

# Install Freqtrade first (brings its deps; avoid TA-Lib by not installing ta-lib system lib)
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -r /app/requirements.extra.txt

# Copy the rest of the project
COPY . /app

# Default command shows CLI help
CMD ["python", "rl_trader.py", "--help"]