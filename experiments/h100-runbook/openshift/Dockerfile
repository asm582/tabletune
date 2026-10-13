# Base image for the benchmark pod. OpenShift runs pods as an arbitrary non-root UID, so
# everything that needs apt is installed here; Python, vLLM and the package are installed
# later onto the persistent volume by scripts/01_setup.sh.
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential procps util-linux tmux less jq \
    && rm -rf /var/lib/apt/lists/*
ENV HOME=/work
WORKDIR /work
CMD ["sleep", "infinity"]
