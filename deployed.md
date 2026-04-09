# Deployment

## Current Deployment Status

**Local Deployment Only** - This application is designed for local/internal deployment.

### Running the Application

```bash
cd Project_Final
python -m app.main
```

Access at: http://localhost:5000

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install langchain-groq
COPY . .
CMD ["python", "-m", "app.main"]
```

### Build and Run

```bash
docker build -t policy-rag .
docker run -p 5000:5000 -e GROQ_API_KEY=your-key policy-rag
```

## Production Best Practices

- Use HTTPS via Nginx reverse proxy
- Store API keys in environment variables
- Never commit .env file to Git
- Use Gunicorn instead of Flask dev server

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| GROQ_API_KEY | Yes | Groq API key |
| PORT | No | Override default port |

## Health Check

```bash
curl http://localhost:5000/health
```

Expected response: {"indexed": true, "status": "healthy", "vector_store_ready": true}