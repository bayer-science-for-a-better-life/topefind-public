FROM python:3.8-slim-buster

# Get pip and get build tools
RUN apt-get update -y && apt-get install -y \
    python3-pip

# Copy and install
COPY requirements.txt /app/requirements.txt
WORKDIR /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app
RUN cd /app && make install && mkdir /data/

ENTRYPOINT ["python", "/app/cli.py"]