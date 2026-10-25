#!/bin/bash
echo "Начинается установка AdimAI"

pip install -q torch transformers accelerate bitsandbytes huggingface_hub

curl -sO https://raw.githubusercontent.com/DmitryStarikov-tech/adimai/main/runproject.py

python run_project.py
