FROM python:3.13-slim
WORKDIR /bot
COPY code_ver10.py mechfight.py /bot/
RUN python3 -c "from pathlib import Path; import hashlib; assert hashlib.sha256(Path('code_ver10.py').read_bytes().replace(bytes([13,10]), bytes([10]))).hexdigest() == '279f1787853548a16ffda8f1ee3565148602e419f8555b7b44e8296ccc2916d2', 'Wrong bot file: expected Darwin Range Guard'"
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/code_ver10.py"]
