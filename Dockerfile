FROM ubuntu:16.04

MAINTAINER Georgy Marrero "georgymarrero@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    pip install --upgrade pip && \
    apt-get install -y libgtk2.0-dev

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN pip install --no-cache-dir tensorflow==1.15.0

RUN pip install gunicorn[gevent]

EXPOSE 8080

ENTRYPOINT [ "bash" ]

CMD ["start.sh"]
