FROM python:3.13-slim
WORKDIR /bot
COPY code_ver10.py mechfight.py /bot/
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/code_ver10.py"]
