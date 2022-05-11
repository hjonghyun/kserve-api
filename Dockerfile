FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt # Write Flask in this file
EXPOSE 5000 

ENTRYPOINT [ "./gunicorn.sh" ]