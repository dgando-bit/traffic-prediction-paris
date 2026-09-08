FROM python:3.12-slim

WORKDIR /app

COPY packages/shared /app/packages/shared
COPY packages/traffic_prediction /app/packages/traffic_prediction
COPY services/api /app/services/api

RUN pip install --no-cache-dir \
    /app/packages/shared \
    /app/packages/traffic_prediction \
    /app/services/api

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]