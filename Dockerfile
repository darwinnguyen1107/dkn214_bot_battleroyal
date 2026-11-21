FROM python:3.13-slim
WORKDIR /bot
COPY . /bot/
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/Examplev8.py"]
