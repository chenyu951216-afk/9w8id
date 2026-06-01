FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

EXPOSE 8080

CMD ["python", "-m", "crypto_ict_bot", "ui", "--host", "0.0.0.0", "--no-browser"]
