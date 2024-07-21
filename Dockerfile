FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY requirements-app.txt .

RUN apt-get update \
    && apt-get install -y gcc g++ libffi-dev libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir -r requirements-app.txt

RUN apt-get remove -y gcc g++ \
    && apt-get autoremove -y \
    && apt-get clean

COPY . .

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.wsgi:app"]