FROM python:3.11-slim
# Start from a clean Linux container with Python installed

WORKDIR /app
# Set the working directory inside the container

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install all Python packages (fastapi, sentence-transformers, etc.)

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# Download the 90MB AI model NOW at build time
# so it's baked into the image — never needs internet at runtime

COPY contextcrunch ./contextcrunch
# Copy your Python library code into the container

COPY backend/ .
# Copy main.py, requirements.txt into the container

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# Start the FastAPI server when the container runs