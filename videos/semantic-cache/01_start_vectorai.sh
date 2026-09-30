#!/usr/bin/env bash
# VectorAI DB, local. Ports: 6574 gRPC (the client), 6575 web UI.
set -e

docker run -d --name vectorai \
  -v ./local_data:/var/lib/actian-vectorai \
  -p 6573-6575:6573-6575 \
  -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
  actian/vectorai:latest

docker ps --filter name=vectorai
