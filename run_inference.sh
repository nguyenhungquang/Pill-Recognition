#!/bin/bash
docker build -t base_image .
docker run \
    --name inference_test \
    --mount type=bind,source="$(pwd)"/VAIPE,target=/app/VAIPE \
    --rm \
    --gpus all \
    base_image \
    run_inference.sh