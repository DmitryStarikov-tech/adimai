#!/bin/bash
echo "Начинается установка AdimAI"

pip install -q torch transformers accelerate bitsandbytes huggingface_hub pandas matplotlib

curl -sO https://raw.githubusercontent.com/DmitryStarikov-tech/adimai/main/run_project.py
curl -sO https://raw.githubusercontent.com/DmitryStarikov-tech/adimai/main/sandbox.py

python run_project.py < /dev/tty
