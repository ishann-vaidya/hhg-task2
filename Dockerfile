FROM python:3.12-slim

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase and data indexes
COPY . .

# Expose default port and launch via main.py
EXPOSE 8000
CMD ["python", "main.py"]
