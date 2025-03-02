FROM python:3.9-slim

# Install build tools (gcc, make, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /LALAS

COPY requirements.txt .

# Installing required dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5005

CMD ["python", "main.py"]