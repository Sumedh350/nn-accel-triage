FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps: Python 3.10, Verilator build tools, Node.js
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git autoconf flex bison libfl2 libfl-dev cmake make g++ perl \
    help2man \
    curl ca-certificates \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

# Build Verilator 5.x from source (pinned to v5.020 — satisfies 5.x requirement)
RUN git clone https://github.com/verilator/verilator.git \
      --branch v5.020 --depth 1 /opt/verilator-src \
  && cd /opt/verilator-src \
  && autoconf \
  && ./configure --prefix=/usr/local \
  && make -j$(nproc) \
  && make install \
  && rm -rf /opt/verilator-src

WORKDIR /workspace
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["/bin/bash", "docker/run_pipeline.sh"]
