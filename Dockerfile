FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY . /app

EXPOSE 8000

# default command runs the HF inference script; override as needed
CMD ["python", "hf_inference.py", "--prompt", "Hello from container", "--max-length", "50"]
