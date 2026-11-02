FROM python:3.13-slim

# libgomp1 is CatBoost's OpenMP runtime dependency.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Do the slow first-run work at build time, not on Render's 0.1-CPU free instance:
# byte-compile everything and build matplotlib's font cache.
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/app/.mplcache
RUN python -m compileall -q /usr/local/lib/python3.13 /app \
    && python -c "import matplotlib.pyplot"

ENV PORT=10000
EXPOSE 10000

CMD ["python", "app.py"]
