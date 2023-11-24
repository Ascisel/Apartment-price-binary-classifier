FROM Python:3.11.6

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PATH="/app:${PATH}"

ENV FLASK_APP=flask_app.Python

EXPOSE 8080

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]