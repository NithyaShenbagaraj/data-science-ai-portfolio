FROM python:3.11-slim

WORKDIR /app
COPY 02-deep-learning/agency-revenue-intelligence/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY 02-deep-learning/agency-revenue-intelligence /app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
