# Stage 1: Builder stage
FROM python:3.12-slim AS builder

WORKDIR /code

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    wget \
    ca-certificates \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Install Playwright and its dependencies
RUN pip install --no-cache-dir playwright==1.49.1
RUN python -m playwright install

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /code

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    wget \
    ca-certificates \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and Playwright dependencies
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Reinstall Playwright browsers to ensure correctness
RUN playwright install-deps

# Copy application code
COPY ./app /code/app

EXPOSE 8000

# Run the application
CMD ["sh", "-c", "python -m playwright install-deps && uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload"]
