# backend/Dockerfile
FROM python:3.11-slim

# Avoid Python buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps if needed (minimal for now)
RUN pip install --no-cache-dir --upgrade pip

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Expose port
EXPOSE 8000

# Run FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
