# Image for running the benchmark on Nautilus.
#
#   docker build -t ghcr.io/<you>/rl-benchmark:latest .
#   docker push  ghcr.io/<you>/rl-benchmark:latest
#   python rl_benchmark.py --k8s ghcr.io/<you>/rl-benchmark:latest
#   kubectl apply -f results/k8s
#
# Untested. I have no cluster to submit it to, so treat this as a starting point
# rather than something known to work.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Box2D needs swig and a compiler at build time, and CarRacing needs the SDL
# and GL libraries even when rendering headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        swig build-essential \
        libgl1 libglib2.0-0 libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rl_benchmark.py test_framework.py ./

# Fail fast: if the environment cannot be created, the job should die in seconds
# rather than after an hour of an empty learning curve.
RUN python -c "import rl_benchmark as R; e = R.CarRacingDiscrete(seed=0); e.reset(seed=0); e.step(3); e.close(); print('env ok')"

ENV MPLBACKEND=Agg
ENTRYPOINT ["python", "rl_benchmark.py"]
CMD ["--all", "--steps", "200000"]
