# AdimAI

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-orange)

**AdimAI** — это кастомный ИИ-проект, представляющий собой гибридную архитектуру на базе `Qwen2.5-Coder-1.5B`. Проект демонстрирует интеграцию классического дообучения с авторскими архитектурными решениями: программируемой латентной памятью (PLE) и роутерами условной глубины (Early-Exit).


### Динамическая система скиллов
Модель поддерживает переключение скиллов прямо во время работы. Это позволяет на лету менять системный промпт и жестко задавать правила генерации кода под конкретные задачи, минимизируя ошибки синтаксиса.

**Доступные команды в терминале:**
* `/skill default` — Стандартный режим.
* `/skill analyst` — Режим дата-саентиста. Модель получает жесткие инструкции. Идеально для работы с CSV в песочнице.

*Вы можете легко добавлять собственные скиллы, расширяя словарь `system_prompts` в файле `run_project.py`!*

## Быстрый старт (Установка в 1 клик)
Запустите эту команду в терминале (macOS/Linux), чтобы автоматически скачать веса, развернуть окружение и запустить интерактивный чат:

curl -s [https://raw.githubusercontent.com/DmitryStarikov-tech/adimai/main/start.sh](https://raw.githubusercontent.com/DmitryStarikov-tech/adimai/main/start.sh) | bash

