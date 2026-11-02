FROM python:3.11-slim

# System dependencies required by WeasyPrint (Cairo, Pango, GDK-PixBuf)
# and psycopg2 (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # WeasyPrint deps
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    # psycopg2 dep
    libpq-dev \
    # Font support
    fonts-liberation \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py database.py models.py utils.py ./

# Copy ML model and data assets
# lr_pipeline.pkl     — trained scikit-learn pipeline
# raw_data.csv        — source data for synthetic sample generation
# students_metadata.json — SDV metadata cache (optional, auto-generated if absent)
COPY lr_pipeline.pkl raw_data.csv ./

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]