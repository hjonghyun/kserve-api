FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app
RUN chmod +x /app/gunicorn.sh
RUN pip install -r requirements.txt # Write Flask in this file
EXPOSE 5000 

ENTRYPOINT [ "sh","/app/gunicorn.sh" ]