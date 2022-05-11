FROM python:3.7-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git

# ENV APP_HOME /app
# WORKDIR $APP_HOME
# ADD transformer_test.py /app/
# ADD requirements.txt /app/
# ADD train_stats.csv /app/

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "transformer_test.py"]