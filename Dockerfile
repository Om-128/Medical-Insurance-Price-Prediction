FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN python -m src.pipeline.train_pipeline
EXPOSE 5000
CMD  ["python", "app.py"]