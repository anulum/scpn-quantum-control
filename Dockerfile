FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/
COPY examples/ examples/
COPY notebooks/ notebooks/

CMD ["pytest", "tests/", "-v", "--tb=short"]
