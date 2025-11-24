
FROM node:18-alpine as build-step


WORKDIR /app/frontend


COPY frontend/package.json frontend/package-lock.json ./


RUN npm install


COPY frontend/ ./


RUN npm run build



FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY --from=build-step /app/frontend/build /app/static_ui


COPY . .


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]