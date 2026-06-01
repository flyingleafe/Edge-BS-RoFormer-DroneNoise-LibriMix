#!/usr/bin/env bash
set -e
echo "Hello from postdoc!"
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
echo "Sleeping for 10 seconds..."
sleep 10
echo "Done!"
