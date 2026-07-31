FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY superkart_rf_sales_forecast_v1_0.joblib .

EXPOSE 7860

CMD ["python", "app.py"]