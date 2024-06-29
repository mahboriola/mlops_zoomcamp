FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY score.py /app
COPY Pipfile /app
COPY Pipfile.lock /app

RUN pip install pipenv
RUN pipenv install --python 3.10 --system --deploy --ignore-pipfile
RUN mkdir output