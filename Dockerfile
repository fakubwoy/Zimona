FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure uploads dir exists at build time; on Railway this gets shadowed
# by the mounted volume so build-time contents don't matter.
RUN mkdir -p static/uploads/categories

EXPOSE 80

# Use gunicorn in production. Workers=2 keeps RAM low on Railway's free tier.
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "2", "--timeout", "120", "--preload", "app:app"]