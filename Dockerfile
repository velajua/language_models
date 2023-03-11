FROM python:3.9-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt-get update && apt-get install -y libgomp1
# RUN apt-get install gcc
RUN pip install Flask gunicorn
RUN pip install --no-cache-dir Cython
# RUN pip install --upgrade pip | pip install -r requirements.txt
RUN pip install -Ur requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 7200 app:app
